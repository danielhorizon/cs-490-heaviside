

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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


EPS = 1e-7
_IRIS_DATA_PATH = "../data/iris.csv"
_MODELS_PATH = "/app/timeseries/multiclass_src/models/iris"

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
        self.X = torch.from_numpy(ds_split['X']).float()
        self.y = torch.from_numpy(ds_split['y']).float()

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


def load_model(model_name):
    model_path = "/".join([_MODELS_PATH, model_name])
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = torch.load(model_path).to("cuda:3")
    model.eval()
    return model


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


def get_metrics(device, model_name, batch_size, seed, output_file):
    # LOAD IN TEST LOADER
    set_seed(seed)
    data_splits = load_iris(seed=seed)
    train_loader, val_loader, test_loader = create_loaders(data_splits, batch_size=batch_size, seed=seed)


    # LOAD iN MODEL
    model = load_model(model_name)

    # EVALUATION
    model.eval()
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

    eval_json = {
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
            final_test_dxn = [0, 0, 0]
            test_preds, test_labels = [], []
            for i, (inputs, labels) in enumerate(test_loader):
                # updating distribution of labels.
                labels_list = labels.numpy()
                for label in labels_list:
                    final_test_dxn[int(label)] += 1

                # stacking onto tensors.
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.type(torch.int64)

                # passing it through our finalized model.
                output = model(inputs)
                labels = torch.zeros(len(labels), 3).to(device).scatter_(
                    1, labels.unsqueeze(1), 1.).to(device)

                pred_arr = output.detach().cpu().numpy()
                label_arr = labels.detach().cpu().numpy()

                # appending results.
                test_preds.append(pred_arr)
                test_labels.append(label_arr)

            test_preds = torch.tensor(test_preds[0])
            test_labels = torch.tensor(test_labels[0])

            class_f1s, mean_f1, precisions, recalls = evaluation_f1(
                device=device, y_labels=test_labels, y_preds=test_preds, threshold=tau)

            tau = str(tau)
            eval_json[tau]['class_f1s'] = class_f1s.numpy().tolist()
            eval_json[tau]['mean_f1'] = mean_f1.item()
            eval_json[tau]['eval_dxn'] = final_test_dxn
            eval_json[tau]['class_precisions'] = precisions.numpy().tolist()
            eval_json[tau]['class_recalls'] = recalls.numpy().tolist()
    eval_json['run_name'] = model_name
    record_results(best_test=eval_json,
                   results_path="/app/timeseries/multiclass_src/results/iris", 
                   output_file=output_file)
    return eval_json


if __name__ == '__main__':
    ## batch size = 128 
    af1_batch_128 = [
        "20201215-best_model-128-approx-f1-0.pth", 
        "20201215-best_model-128-approx-f1-1.pth", 
        "20201215-best_model-128-approx-f1-2.pth", 
        "20201215-best_model-128-approx-f1-3.pth", 
        "20201215-best_model-128-approx-f1-4.pth"
    ]
    for model in af1_batch_128:
        get_metrics(device="cuda:3", model_name=model, batch_size=128, seed=11, output_file="128_f1_results.json")
    
    baseline_batch_128 = [
        "20201215-best_model-128-baseline-ce-0.pth", 
        "20201215-best_model-128-baseline-ce-1.pth", 
        "20201215-best_model-128-baseline-ce-2.pth", 
        "20201215-best_model-128-baseline-ce-3.pth", 
        "20201215-best_model-128-baseline-ce-4.pth"
    ]
    for model in baseline_batch_128:
        get_metrics(device="cuda:3", model_name=model, batch_size=128,
                    seed=11, output_file="128_ce_results.json")

    ## batch size = 256 
    af1_batch_256 = [
        "20201215-best_model-256-approx-f1-0.pth",
        "20201215-best_model-256-approx-f1-1.pth",
        "20201215-best_model-256-approx-f1-2.pth",
        "20201215-best_model-256-approx-f1-3.pth", 
        "20201215-best_model-256-approx-f1-4.pth"
    ]
    for model in af1_batch_256:
        get_metrics(device="cuda:3", model_name=model, batch_size=256, seed=11, output_file="256_f1_results.json")

    baseline_batch_256 = [
        "20201215-best_model-256-baseline-ce-0.pth",
        "20201215-best_model-256-baseline-ce-1.pth",
        "20201215-best_model-256-baseline-ce-2.pth",
        "20201215-best_model-256-baseline-ce-3.pth",
        "20201215-best_model-256-baseline-ce-4.pth"
    ]
    for model in baseline_batch_256:
        get_metrics(device="cuda:3", model_name=model, batch_size=256,
                    seed=11, output_file="256_ce_results.json")

            

