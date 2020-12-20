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
_MODELS_PATH = "/app/timeseries/multiclass_src/models/iris-poc"
_IRIS_DATA_PATH = "../data/iris.csv"

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
    
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = torch.load(model_path).to("cuda:3")
    return model


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


def get_metrics(device, batch_size, seed, results_path, models_path, output_file, run_number):
    # LOAD IN TEST LOADER 
    set_seed(seed)
    data_splits = load_iris(seed=seed)
    _, _, test_loader = create_loaders(
        data_splits, batch_size=batch_size, seed=seed)

    temp_json = {
        "0.1": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.125": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.2": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.3": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.4": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.5": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.6": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.7": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.8": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
        "0.9": {
            "0.1": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.2": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.3": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.4": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.45": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.5": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.55": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.6": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.7": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.8": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            },
            "0.9": {
                "1": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "2": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
                "3": {"class_f1s": None, 'class_precisions': None, 'class_recalls': None},
            }
        },
    }

    # EVALUATION
    results_json = {   
        "0.1": {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.125":    {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.2":  {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.3":   {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.4":    {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.5":  {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.6":   {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.7":  {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.8":    {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
        "0.9":   {
            "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
            "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
        },
    }

    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    trained_taus = ["0.1", "0.125", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

    with torch.no_grad():    
        # loop through every single class 
        for x in range(len(trained_taus)): 
            # looping through every single train tau 
            for i in range(1,4): 
                # load in the model 
                print("LOADING IN Train Tau: {}, Class {}".format(trained_taus[x], i))
                model_name = "class-{}-best-model-poc-af1-{}-{}.pth".format(i, trained_taus[x], run_number)
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
                        labels = torch.zeros(len(labels), 3).to(device).scatter_(1, labels.unsqueeze(1), 1.).to(device)

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
    
    ## recording results
    # record_results(best_test=temp_json,
    #                results_path=results_path,
    #                output_file="temp.json")

    ## need to process results 
    for i in range(len(trained_taus)):
        for j in range(len(test_thresholds)): 
            # loop through every single class 
            class_f1s, class_precisions, class_recalls = [None]*3, [None]*3, [None]*3 
            for k in range(1,4):
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


if __name__ == '__main__':
    trained_taus = ["0.1", "0.125", "0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    for i in range(5): 
        get_metrics(device="cuda:3", batch_size=256, seed=11,
                            results_path="/app/timeseries/multiclass_src/results/iris/",
                            models_path="/app/timeseries/multiclass_src/models/iris-poc",
                            output_file="poc_agg.json", 
                            run_number=i)


# def process_results():
#     results_json = {
#         "0.1": {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.125":    {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.2":   {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.3":   {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.4":   {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.5":   {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.6":  {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.7":  {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.8":   {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#         "0.9":   {
#             "0.1": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.2": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.3": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.4": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.45": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.5": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.55": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.6": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.7": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.8": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#             "0.9": {"class_f1s": [None]*3, 'class_precisions': [None]*3, 'class_recalls': [None]*3},
#         },
#     }

#     with open("/app/timeseries/multiclass_src/results/poc/temp.json", "r+") as f:
#         temp_json = json.load(f)
#     temp_json = temp_json[0]

#     test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
#     trained_taus = ["0.1", "0.125", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
#     ## need to process results 
#     for i in range(len(trained_taus)):
#         for j in range(len(test_thresholds)): 
#             # loop through every single class 
#             class_f1s, class_precisions, class_recalls = [None]*3, [None]*3, [None]*3 
#             for k in range(1,4):
#                 class_f1s[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_f1s'][k-1]
#                 class_precisions[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_precisions'][k-1]
#                 class_recalls[k-1] = temp_json[trained_taus[i]][str(test_thresholds[j])][str(k)]['class_recalls'][k-1]

#             results_json[trained_taus[i]][str(test_thresholds[j])]['class_f1s'] = class_f1s
#             results_json[trained_taus[i]][str(test_thresholds[j])]['class_precisions'] = class_precisions
#             results_json[trained_taus[i]][str(test_thresholds[j])]['class_recalls'] = class_recalls
    
#     record_results(best_test=results_json,
#                    results_path="/app/timeseries/multiclass_src/results/iris/poc/",
#                    output_file="poc_results.json")



