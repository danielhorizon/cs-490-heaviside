

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
from mnist_helper import load_mnist_imbalanced, load_balanced_data


EPS = 1e-7


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


# https://nextjournal.com/gkoehler/pytorch-mnist
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
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
    model = torch.load(model_path).to("cuda:0")
    return model


def get_test_loader(batch_size, seed):
    data_splits = load_mnist_imbalanced(seed)
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


def get_metrics(device, batch_size, seed, results_path, models_path, models_list, output_file, imbalanced):
    # LOAD IN TEST LOADER 
    if imbalanced: 
        _, _, test_loader = get_test_loader(batch_size=batch_size, seed=seed)
    else: 
        _, _, test_loader = load_balanced_data(seed=seed, batch_size=batch_size)

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
        for x in range(len(models_list)): 
            print("Loading in model: {}".format(models_list[x]))
            model = load_model(models_path, models_list[x])
            model.eval()
            
            for tau in test_thresholds:
                # go through all the thresholds, and test them out again.
                test_preds, test_labels = [], []
                for i, (inputs, labels) in enumerate(test_loader):
                    # stacking onto tensors.
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # passing it through our finalized model.
                    output = model(inputs)
                    labels = torch.zeros(len(labels), 10).to(device).scatter_(
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
                eval_json[tau]['class_precisions'] = precisions.numpy().tolist()
                eval_json[tau]['class_recalls'] = recalls.numpy().tolist()
    record_results(best_test=eval_json,
                   results_path=results_path, 
                   output_file=output_file)
    return eval_json


if __name__ == '__main__':
    # approx_f1_models = [
    #     "model-1024-approx-f1-imb-0.pth",
    #     "model-1024-approx-f1-imb-1.pth",
    #     "model-1024-approx-f1-imb-2.pth",
    # ]
    # get_metrics(device="cuda:0", batch_size=1024, seed=1,
    #             results_path="/app/timeseries/multiclass_src/results/mnist",
    #             models_path="/app/timeseries/multiclass_src/models/mnist",
    #             models_list=approx_f1_models,
    #             output_file="20201220-mnist-imb-results.json",
    #             imbalanced=True)

    # ce_models = [
    #     "model-1024-approx-f1-reg-0.pth",
    #     "model-1024-approx-f1-reg-1.pth",
    #     "model-1024-approx-f1-reg-2.pth",
    # ]
    # get_metrics(device="cuda:0", batch_size=1024, seed=1,
    #             results_path="/app/timeseries/multiclass_src/results/mnist",
    #             models_path="/app/timeseries/multiclass_src/models/mnist",
    #             models_list=ce_models,
    #             output_file="20201220-mnist-reg-results.json",
    #             imbalanced=False)

    approx_f1_models = [
        "model-1024-baseline-ce-imb-0.pth",
        "model-1024-baseline-ce-imb-1.pth",
        "model-1024-baseline-ce-imb-2.pth",
    ]
    get_metrics(device="cuda:0", batch_size=1024, seed=1,
                results_path="/app/timeseries/multiclass_src/results/mnist",
                models_path="/app/timeseries/multiclass_src/models/mnist",
                models_list=approx_f1_models,
                output_file="20201220-ce-imb.json",
                imbalanced=True)

    ce_models = [
        "model-1024-baseline-ce-reg-0.pth",
        "model-1024-baseline-ce-reg-1.pth",
        "model-1024-baseline-ce-reg-2.pth",
    ]
    get_metrics(device="cuda:0", batch_size=1024, seed=1,
                results_path="/app/timeseries/multiclass_src/results/mnist",
                models_path="/app/timeseries/multiclass_src/models/mnist",
                models_list=ce_models,
                output_file="20201220-ce-reg.json",
                imbalanced=False)

    # search_tau_models = [
    #     "20201211-best_model-v3-searchtau-e5-1024-0.pth", 
    #     "20201211-best_model-v3-searchtau-e5-1024-1.pth", 
    #     "20201211-best_model-v3-searchtau-e5-1024-2.pth"
    # ]
    # for stau_model in search_tau_models:
    #     get_metrics(device="cuda:3", model_name=stau_model,
    #                 batch_size=1024, seed=11, output_file="20201208_e5_agg_results.json")

    # search_tau_models = [
    #     "20201211-best_model-v2-searchtau-e10-1024-0.pth",
    #     "20201211-best_model-v2-searchtau-e10-1024-1.pth",
    #     "20201211-best_model-v2-searchtau-e10-1024-2.pth"
    # ]
    # for stau_model in search_tau_models:
    #     get_metrics(device="cuda:3", model_name=stau_model,
    #                 batch_size=1024, seed=11, output_file="20201208_e10_agg_results.json")
