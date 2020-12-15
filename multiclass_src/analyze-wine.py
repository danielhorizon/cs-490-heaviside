

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
_WHITE_WINE = "../data/winequality-white.csv"
_RED_WINE = "../data/winequality-red.csv"
_MODELS_PATH = "/app/timeseries/multiclass_src/models/wine"


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
    train_df, test_df = train_test_split(
        raw_df, test_size=0.2, shuffle=shuffle)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, shuffle=shuffle)

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


def get_metrics(device, model_name, batch_size, seed, output_file):
    # LOAD IN TEST LOADER
    set_seed(seed)
    data_splits = load_white_wine(seed=seed)
    _, _, test_loader = create_loaders(data_splits, batch_size=batch_size, seed=seed)


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
            final_test_dxn = [0, 0, 0, 0]
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
                labels = torch.zeros(len(labels), 4).to(device).scatter_(
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
                   results_path="/app/timeseries/multiclass_src/results/wine", 
                   output_file=output_file)
    return eval_json


if __name__ == '__main__':
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

    ## batch size = 512
    af1_batch_512 = [
        "20201215-best_model-512-approx-f1-0.pth",
        "20201215-best_model-512-approx-f1-1.pth",
        "20201215-best_model-512-approx-f1-2.pth",
        "20201215-best_model-512-approx-f1-3.pth",
        "20201215-best_model-512-approx-f1-4.pth"
    ]
    for model in af1_batch_512:
        get_metrics(device="cuda:3", model_name=model, batch_size=512,
                    seed=11, output_file="512_f1_results.json")

    baseline_batch_512 = [
        "20201215-best_model-512-baseline-ce-0.pth",
        "20201215-best_model-512-baseline-ce-1.pth",
        "20201215-best_model-512-baseline-ce-2.pth",
        "20201215-best_model-512-baseline-ce-3.pth",
        "20201215-best_model-512-baseline-ce-4.pth"
    ]
    for model in baseline_batch_512:
        get_metrics(device="cuda:3", model_name=model, batch_size=512,
                    seed=11, output_file="512_ce_results.json")
