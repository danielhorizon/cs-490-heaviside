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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# CUSTOM IMPORT
from download_cifar import *


EPS = 1e-7

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


def load_model(models_path, model_name):
    model_path = "/".join([models_path, model_name])
    
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = torch.load(model_path).to("cuda:3")
    return model


def get_test_loader(batch_size, seed):
    data_splits = load_imb_data_v2(seed)
    train_set = Dataset(data_splits['train'])
    validation_set = Dataset(data_splits['val'])
    test_set = Dataset(data_splits['test'])

    data_params = {'batch_size': batch_size, 'shuffle': True,
                   'num_workers': 0, 'worker_init_fn': np.random.seed(seed)}
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


# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
def load_data_v2(shuffle=True, batch_size=None, seed=None):
    torch.manual_seed(seed)

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms for validation and train.
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # loading in dataset.
    train_dataset = CIFAR10(train=True, download=True,
                            root="../data", transform=train_transform)
    valid_dataset = CIFAR10(train=True, download=True,
                            root="../data", transform=valid_transform)
    # need to transform the test according to the train.
    test_dataset = CIFAR10(train=False, download=True,
                           root="../data", transform=train_transform)

    print("Train Size: {}, Test Size: {}, Valid Size: {}".format(
        len(train_dataset), len(test_dataset), len(valid_dataset)))

    # spliiting into validation/train/test.
    num_train = len(train_dataset)
    indices = list(range(num_train))
    valid_size = 0.10
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print("Train Size:{} Valid Size: {}".format(len(train_idx), len(valid_idx)))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    return train_loader, valid_loader, test_loader


def get_metrics(device, batch_size, seed, results_path, models_path, output_file, imbalanced, run_number):
    # LOAD IN TEST LOADER 
    if imbalanced:
        _, _, test_loader  = get_test_loader(batch_size=batch_size, seed=seed)
    else: 
        _, _, test_loader = load_data_v2(batch_size=batch_size, seed=seed)

    temp_json = {
        "0.1": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.125": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.2": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.3": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.4": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.5": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.6": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.7": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.8": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.9": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "4": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "5": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "6": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "7": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "8": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "9": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "10": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        }
    }

    # EVALUATION
    results_json = {   
        "0.1": {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.125":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.2":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.3":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.4":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.5":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.6":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.7":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.8":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.9":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
    }

    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    trained_taus = ["0.1", "0.125", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

    with torch.no_grad():    
        # loop through every single class 
        for x in range(len(trained_taus)): 
            # looping through every single train tau 
            for i in range(1,11): 
                # load in the model 
                print("LOADING IN Train Tau: {}, Class {}".format(trained_taus[x], i))
                model_name = "class-{}-best-model-poc-af1-imb-{}-{}.pth".format(i, trained_taus[x], run_number)
                model = load_model(models_path, model_name)
                model.eval()
                
                # looping through evaluation taus 
                for tau in test_thresholds:
                    test_preds, test_labels = [], []
                    for batch, (inputs, labels) in enumerate(test_loader):
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

                    # this is for class 1's 
                    class_f1s, mean_f1, precisions, recalls = evaluation_f1(device=device, y_labels=test_labels, y_preds=test_preds, threshold=tau)

                    tau = str(tau)
                    # train taus -> class -> 
                    temp_json[trained_taus[x]][str(tau)][str(i)]['class_f1s'] = class_f1s.numpy().tolist()
                    temp_json[trained_taus[x]][str(tau)][str(i)]['mean_f1'] = mean_f1.item()
                    temp_json[trained_taus[x]][str(tau)][str(i)]['class_precisions'] = precisions.numpy().tolist()
                    temp_json[trained_taus[x]][str(tau)][str(
                        i)]['class_recalls'] = recalls.numpy().tolist()
    

    ## need to process results 
    for i in range(len(trained_taus)):
        for j in range(len(test_thresholds)): 
            # loop through every single class 
            class_f1s, class_precisions, class_recalls = [None]*10, [None]*10, [None]*10 
            for k in range(1,11):
                class_f1s[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_f1s'][k-1]
                class_precisions[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_precisions'][k-1]
                class_recalls[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_recalls'][k-1]

            results_json[trained_taus[i]][str(test_thresholds[j])]['class_f1s'] = class_f1s
            results_json[trained_taus[i]][str(test_thresholds[j])]['class_precisions'] = class_precisions
            results_json[trained_taus[i]][str(test_thresholds[j])]['class_recalls'] = class_recalls

    ## recording results 
    record_results(best_test=results_json,
                   results_path=results_path, 
                   output_file=output_file)
    return results_json


def process_results():
    results_json = {
        "0.1": {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.125":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.2":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.3":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.4":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.5":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.6":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.7":  {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.8":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
        "0.9":   {
            "0.1": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.2": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.3": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.4": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.45": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.5": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.55": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.6": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.7": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.8": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
            "0.9": {"class_f1s": [None]*10, 'class_precisions': [None]*10, 'class_recalls': [None]*10},
        },
    }

    with open("/app/timeseries/multiclass_src/results/poc/temp.json", "r+") as f:
        temp_json = json.load(f)
    temp_json = temp_json[0]

    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    trained_taus = ["0.1", "0.125", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    ## need to process results 
    for i in range(len(trained_taus)):
        for j in range(len(test_thresholds)): 
            # loop through every single class 
            class_f1s, class_precisions, class_recalls = [None]*10, [None]*10, [None]*10 
            for k in range(1,11):
                class_f1s[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_f1s'][k-1]
                class_precisions[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_precisions'][k-1]
                class_recalls[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_recalls'][k-1]

            results_json[trained_taus[i]][str(test_thresholds[j])]['class_f1s'] = class_f1s
            results_json[trained_taus[i]][str(test_thresholds[j])]['class_precisions'] = class_precisions
            results_json[trained_taus[i]][str(test_thresholds[j])]['class_recalls'] = class_recalls
    
    record_results(best_test=results_json,
                   results_path="/app/timeseries/multiclass_src/results/poc/",
                   output_file="poc_results.json")


if __name__ == '__main__':
    trained_taus = ["0.1", "0.125", "0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    for i in range(3): 
        get_metrics(device="cuda:3", batch_size=1024, seed=11,
                            results_path="/app/timeseries/multiclass_src/results/poc/cifar/",
                            models_path="/app/timeseries/multiclass_src/models/cifar-10-poc",
                            output_file="poc_results.json",
                            imbalanced=True, 
                            run_number=i)

