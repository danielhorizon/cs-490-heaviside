import os
import click
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mc_metrics import get_confusion

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# for early stopping.
from pytorchtools import EarlyStopping
from mc_torchconfusion import *
from mc_torchconfusion_weighted import wt_mean_f1_approx_loss_on

from download_cifar import *


'''
# https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Input -> Conv (ReLU) -> MaxPool -> Conv (ReLU) -> MaxPool -> 
    FC (ReLU) -> FC (ReLU) -> FC (Softmax) -> 10 outputs 

Conv: convolution layer, ReLU = activation, MaxPool = pooling layer, FC = fully connected, Softmax

Input: 3x32x32 (3 channels, RGB)

1st Conv: Expects 3 channels, convolves 6 filters each of size 3x5x5 
    Padding = 0, Stride = 0, Output must be 6x28x28 because (32 - 5) + 1 = 28 
    This layer has ((5x5x3) + 1)*6 

MaxPool: 2x2 kernel, stride = 2. 
    Drops size from 6x28x28 -> 6x14x14 

2nd Conv: Expects 6 input channels, convolves 16 filters of size 6x5x5 
    Padding = 0, Stride = 1, output becomes 16x10x10 
    This is because (14-5) + 1 = 10. 
    Layer has ((5x5x6) + 1)x16 = 2416 parameters

1st FCL: 
    The output from the final max pooling layer needs to be flattened so we can connect 
    it to a FC layer. Uses ReLU for activation, and has 120 nodes. 
    ((16x5x5) + 1) x 120 = 48120 parameters 

2nd FCL: 
    Connected to another fully connected layer with 84 nodes, using ReLU as an activation function
    This needs (120 + 1)*84 = 10164 parameters 

Output: 
    Uses softmax and is made up of 10 nodes, one for each category in CIFAR. 
    Requires (84 + 1)*10 = 850 parameters
'''


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


def _check_freq(x):
    return np.array(np.unique(x, return_counts=True)).T


def _show_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
def load_data_v2(shuffle=True, seed=None):
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

    batch_size = 1024
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        # THIS WAS JUST FIXED?!?!?!?
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    return train_loader, valid_loader, test_loader


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_imbalanced_data(seed):
    data_splits = load_imb_data(seed)
    train_set = Dataset(data_splits['train'])
    validation_set = Dataset(data_splits['val'])
    test_set = Dataset(data_splits['test'])

    data_params = {'batch_size': 1024, 'shuffle': True,
                   'num_workers': 1, 'worker_init_fn': np.random.seed(seed)}
    set_seed(seed)
    train_loader = DataLoader(train_set, **data_params)
    set_seed(seed)
    val_loader = DataLoader(validation_set, **data_params)
    set_seed(seed)
    test_loader = DataLoader(test_set, **data_params)
    return train_loader, val_loader, test_loader


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


def get_proportions(arr):
    total = sum(arr)
    for i in range(len(arr)):
        arr[i] = arr[i]/total
    return arr


def evaluation_f1(device, y_labels=None, y_preds=None, threshold=None):
    classes = len(y_labels[0])
    mean_f1s = torch.zeros(classes, dtype=torch.float32)

    '''
    y_labels = tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    y_preds = tensor([[0.0981, 0.0968, 0.0977, 0.0869, 0.1180, 0.1081, 0.0972, 0.0919, 0.1003, 0.1050]])
    '''

    # print("LABELS:{}".format(y_labels))
    # print("PREDS: {}".format(y_preds))

    for i in range(classes):
        gt_list = torch.Tensor([x[i] for x in y_labels]).to(device)
        pt_list = y_preds[:, i]

        # GT LIST:tensor([0., 0., 1.,  ..., 0., 1., 0.], device='cuda:0')
        # PT LIST: tensor([0.1047, 0.1021, 0.1016,  ..., 0.1004, 0.1035, 0.1009], device='cuda:0', grad_fn= < SelectBackward > )

        # print("GT LIST:{}".format(gt_list))
        # print("PT LIST:{}".format(pt_list))
        # tensor([1., 1., 1.,  ..., 1., 1., 1.])
        pt_list = torch.Tensor([1 if x >= threshold else 0 for x in pt_list])

        tn, fp, fn, tp = confusion_matrix(y_true=gt_list.cpu().numpy(
        ), y_pred=pt_list.cpu().numpy(), labels=[0, 1]).ravel()

        # converting to tensors
        tp, fn, fp, tn = torch.tensor([tp]).to(device), torch.tensor([fn]).to(
            device), torch.tensor([fp]).to(device), torch.tensor([tn]).to(device)
        precision = tp/(tp+fp+EPS)
        recall = tp/(tp+fn+EPS)
        temp_f1 = torch.mean(2 * (precision * recall) /
                             (precision + recall + EPS))
        mean_f1s[i] = temp_f1

    # return class wise f1, and the mean of the f1s.
    return mean_f1s, mean_f1s.mean()


def train_cifar(loss_metric=None, epochs=None, imbalanced=None, run_name=None, seed=None, cuda=None):
    using_gpu = False
    if torch.cuda.is_available():
        print("device = cuda")
        print(type(cuda))
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
        using_gpu = True
    else:
        print("device = cpu")
        device = "cpu"

    set_seed(seed)
    train_dxn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_dxn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    valid_dxn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        "final_test_dxn": None,
        "seed": seed
    }

    # setting seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # loading in data
    if imbalanced:
        train_loader, val_loader, test_loader = load_imbalanced_data(seed=seed)
        best_test['imbalanced'] = True
    else:
        train_loader, val_loader, test_loader = load_data_v2(
            shuffle=True, seed=seed)

    model = Net().to(device)
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test['learning_rate'] = learning_rate

    # setting up tensorboard
    if run_name:
        experiment_name = run_name
        tensorboard_path = "/".join(["tensorboard", "cifar_runs", experiment_name])
        writer = SummaryWriter(tensorboard_path)

    # criterion
    approx = False
    if loss_metric == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_metric == "approx-f1":
        approx = True
        criterion = mean_f1_approx_loss_on(device=device)
    elif loss_metric == "approx-f1-wt":
        approx = True
        criterion = wt_mean_f1_approx_loss_on(device=device)
    # elif loss_metric == "approx-acc":
    #     approx = True
    #     criterion = mean_accuracy_approx_loss_on(device=device)
    # elif loss_metric == "approx-auroc":
    #     approx = True
    #     criterion = mean_auroc_approx_loss_on(device=device)
    else:
        raise RuntimeError("Unknown loss {}".format(loss_metric))

    # ----- TRAINING, TESTING, VALIDATION -----
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        accs, microf1s, macrof1s, wf1s = [], [], [], []
        micro_prs, macro_prs, weighted_prs = [], [], []
        micro_recalls, macro_recalls, weighted_recalls = [], [], []
        class_f1_scores = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        class_precision = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        class_recall = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

        ss_class_tp = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        ss_class_fn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        ss_class_fp = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        ss_class_tn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        ss_class_pr = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        ss_class_re = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        ss_class_f1 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        ss_class_acc = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

        if epoch == 0:
            print("--- MODEL PARAMS ---")
            for param in model.parameters():
                print(param.data[1])
                print(param.data[1].shape)
                break

        if epoch != 0:
            # going over in batches of 1024
            for i, (inputs, labels) in enumerate(train_loader):
                # for class distribution - loop through and add
                labels_list = labels.numpy()
                for label in labels_list:
                    train_dxn[label] += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize - batchsize * 10
                output = model(inputs)

                if not approx:
                    loss = criterion(output, labels)
                else:
                    train_labels = torch.zeros(len(labels), 10).to(device).scatter_(
                        1, labels.unsqueeze(1), 1.).to(device)
                    output = output.to(device)

                    loss, hclass_tp, hclass_fn, hclass_fp, hclass_tn, hclass_pr, hclass_re, hclass_f1, hclass_acc = criterion(
                        y_labels=train_labels, y_preds=output)

                losses.append(loss)
                loss.backward()
                optimizer.step()

                # print statistics; every 2000 mini-batches
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                # storing soft-set-metrics
                if approx:
                    for i in range(10):
                        ss_class_tp[i].append(hclass_tp[i])
                        ss_class_fn[i].append(hclass_fn[i])
                        ss_class_fp[i].append(hclass_fp[i])
                        ss_class_tn[i].append(hclass_tn[i])
                        ss_class_pr[i].append(hclass_pr[i])
                        ss_class_re[i].append(hclass_re[i])
                        ss_class_f1[i].append(hclass_f1[i])
                        ss_class_acc[i].append(hclass_acc[i])

                # check prediction
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

                class_f1s = f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)
                class_re = recall_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)
                class_pr = precision_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average=None)

                for i in range(len(class_f1s)):
                    class_f1_scores[i].append(class_f1s[i])
                    class_precision[i].append(class_pr[i])
                    class_recall[i].append(class_re[i])

            m_loss = torch.mean(torch.stack(losses)) if using_gpu else np.array(
                [x.item for x in losses]).mean()
            m_accs = np.array(accs).mean()
            m_weightedf1s = np.array(microf1s).mean()
            m_microf1s = np.array(microf1s).mean()
            m_macrof1s = np.array(macrof1s).mean()
            print("Train - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f}| Macro F1: {:.3f}".format(
                epoch, m_accs, m_weightedf1s, m_microf1s, m_macrof1s)
            )
            if run_name:
                writer.add_scalar("loss", m_loss, epoch)
                writer.add_scalar("train/accuracy", m_accs, epoch)
                writer.add_scalar("train/w-f1", m_weightedf1s, epoch)
                writer.add_scalar("train/micro-f1", m_microf1s, epoch)
                writer.add_scalar("train/macro-f1", m_macrof1s, epoch)
                writer.add_scalar("train/w-recall", np.array(weighted_recalls).mean(), epoch)
                writer.add_scalar("train/micro-recall", np.array(micro_recalls).mean(), epoch)
                writer.add_scalar("train/macro-recall", np.array(macro_recalls).mean(), epoch)
                writer.add_scalar("train/w-precision", np.array(weighted_prs).mean(), epoch)
                writer.add_scalar("train/micro-precision", np.array(micro_prs).mean(), epoch)
                writer.add_scalar("train/macro-precision", np.array(macro_prs).mean(), epoch)

                # adding per-class f1, precision, and recall
                for i in range(10):
                    title = "train/class-" + str(i) + "-f1"
                    writer.add_scalar(title, np.array(class_f1_scores[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-precision"
                    writer.add_scalar(title, np.array(class_precision[i]).mean(), epoch)
                    title = "train/class-" + str(i) + "-recall"
                    writer.add_scalar(title, np.array(class_recall[i]).mean(), epoch)

                    if approx:
                        # adding in softset membership
                        title = "train/class-" + str(i) + "-softset-" + "TP"
                        writer.add_scalar(title, np.array(
                            ss_class_tp[i]).mean(), epoch)
                        title = "train/class-" + str(i) + "-softset-" + "FP"
                        writer.add_scalar(title, np.array(
                            ss_class_fp[i]).mean(), epoch)
                        title = "train/class-" + str(i) + "-softset-" + "FN"
                        writer.add_scalar(title, np.array(
                            ss_class_fn[i]).mean(), epoch)
                        title = "train/class-" + str(i) + "-softset-" + "TN"
                        writer.add_scalar(title, np.array(
                            ss_class_tn[i]).mean(), epoch)
                        title = "train/class-" + \
                            str(i) + "-softset-" + "precision"
                        writer.add_scalar(title, np.array(
                            ss_class_pr[i]).mean(), epoch)
                        title = "train/class-" + \
                            str(i) + "-softset-" + "recall"
                        writer.add_scalar(title, np.array(
                            ss_class_re[i]).mean(), epoch)
                        title = "train/class-" + str(i) + "-softset-" + "f1"
                        writer.add_scalar(title, np.array(
                            ss_class_f1[i]).mean(), epoch)
                        title = "train/class-" + str(i) + "-softset-" + "acc"
                        writer.add_scalar(title, np.array(
                            ss_class_acc[i]).mean(), epoch)

        else:
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
                for i in range(10):
                    title = "train/class-" + str(i) + "-f1"
                    writer.add_scalar(title, 0, epoch)
                    title = "train/class-" + str(i) + "-precision"
                    writer.add_scalar(title, 0, epoch)
                    title = "train/class-" + str(i) + "-recall"
                    writer.add_scalar(title, 0, epoch)

        # ----- TEST SET -----
        # Calculate metrics after going through all the batches
        model.eval()
        test_preds, test_labels = np.array([]), np.array([])
        for i, (inputs, labels) in enumerate(test_loader):
            labels_list = labels.numpy()
            for label in labels_list:
                test_dxn[label] += 1

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

        # add in per-class metrics
        if run_name:
            writer.add_scalar("test/accuracy", test_acc, epoch)
            writer.add_scalar("test/micro-f1", test_f1_micro, epoch)
            writer.add_scalar("test/macro-f1", test_f1_macro, epoch)
            writer.add_scalar("test/w-f1", test_f1_weighted, epoch)
            # adding per-class f1, precision, and recall
            for i in range(10):
                title = "test/class-" + str(i) + "-f1"
                writer.add_scalar(title, np.array(test_class_f1s[i]).mean(), epoch)
                title = "test/class-" + str(i) + "-precision"
                writer.add_scalar(title, np.array(test_class_prs[i]).mean(), epoch)
                title = "test/class-" + str(i) + "-recall"
                writer.add_scalar(title, np.array(test_class_rec[i]).mean(), epoch)

                # adding in per class training
                # get_confusion(gt, pt, class_value=None):
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

        # ----- VALIDATION SET -----
        # Calculate metrics after going through all the batches
        model.eval()
        valid_losses = []
        with torch.no_grad():
            val_preds, val_labels = np.array([]), np.array([])
            ss_class_tp = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            ss_class_fn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            ss_class_fp = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            ss_class_tn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            ss_class_pr = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            ss_class_re = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            ss_class_f1 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            ss_class_acc = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

            for i, (inputs, labels) in enumerate(val_loader):
                labels_list = labels.numpy()
                for label in labels_list:
                    valid_dxn[label] += 1

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
                    valid_labels = torch.zeros(len(labels), 10).to(device).scatter_(
                        1, labels.unsqueeze(1), 1.).to(device)
                    output = output.to(device)
                    curr_val_loss, hclass_tp, hclass_fn, hclass_fp, hclass_tn, hclass_pr, hclass_re, hclass_f1, hclass_acc = criterion(
                        y_labels=valid_labels, y_preds=output)
                else:
                    curr_val_loss = criterion(output, labels)

                valid_losses.append(curr_val_loss.detach().cpu().numpy())

                # storing soft-set-metrics
                if approx:
                    for i in range(10):
                        ss_class_tp[i].append(hclass_tp[i])
                        ss_class_fn[i].append(hclass_fn[i])
                        ss_class_fp[i].append(hclass_fp[i])
                        ss_class_tn[i].append(hclass_tn[i])
                        ss_class_pr[i].append(hclass_pr[i])
                        ss_class_re[i].append(hclass_re[i])
                        ss_class_f1[i].append(hclass_f1[i])
                        ss_class_acc[i].append(hclass_acc[i])

            val_acc = accuracy_score(y_true=val_labels, y_pred=val_preds)
            val_f1_micro = f1_score(y_true=val_labels, y_pred=val_preds, average='micro')
            val_f1_macro = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')
            val_f1_weighted = f1_score(y_true=val_labels, y_pred=val_preds, average='weighted')

            class_val_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average=None)
            class_val_pr = precision_score(y_true=val_labels, y_pred=val_preds, average=None)
            class_val_re = recall_score(y_true=val_labels, y_pred=val_preds, average=None)
            valid_loss = np.mean(valid_losses)

            # add in per-class metrics
            if run_name:
                writer.add_scalar("val/accuracy", val_acc, epoch)
                writer.add_scalar("val/micro-f1", val_f1_micro, epoch)
                writer.add_scalar("val/macro-f1", val_f1_macro, epoch)
                writer.add_scalar("val/w-f1", val_f1_weighted, epoch)

                # adding per-class f1, precision, and recall
                for i in range(10):
                    title = "val/class-" + str(i) + "-f1"
                    writer.add_scalar(title, np.array(
                        class_val_f1[i]).mean(), epoch)
                    title = "val/class-" + str(i) + "-precision"
                    writer.add_scalar(title, np.array(
                        class_val_pr[i]).mean(), epoch)
                    title = "val/class-" + str(i) + "-recall"
                    writer.add_scalar(title, np.array(
                        class_val_re[i]).mean(), epoch)

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

                    if approx:
                        # adding in softset membership
                        title = "val/class-" + str(i) + "-softset-" + "TP"
                        writer.add_scalar(title, np.array(
                            ss_class_tp[i]).mean(), epoch)
                        title = "val/class-" + str(i) + "-softset-" + "FP"
                        writer.add_scalar(title, np.array(
                            ss_class_fp[i]).mean(), epoch)
                        title = "val/class-" + str(i) + "-softset-" + "FN"
                        writer.add_scalar(title, np.array(
                            ss_class_fn[i]).mean(), epoch)
                        title = "val/class-" + str(i) + "-softset-" + "TN"
                        writer.add_scalar(title, np.array(
                            ss_class_tn[i]).mean(), epoch)
                        title = "val/class-" + \
                            str(i) + "-softset-" + "precision"
                        writer.add_scalar(title, np.array(
                            ss_class_pr[i]).mean(), epoch)
                        title = "val/class-" + str(i) + "-softset-" + "recall"
                        writer.add_scalar(title, np.array(
                            ss_class_re[i]).mean(), epoch)
                        title = "val/class-" + str(i) + "-softset-" + "f1"
                        writer.add_scalar(title, np.array(
                            ss_class_f1[i]).mean(), epoch)
                        title = "val/class-" + str(i) + "-softset-" + "acc"
                        writer.add_scalar(title, np.array(
                            ss_class_acc[i]).mean(), epoch)

            # early stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

            print("Val - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Micro F1: {:.3f} | Macro F1: {:.3f}\n".format(
                epoch, val_acc, val_f1_weighted, val_f1_micro, val_f1_macro)
            )
            if epoch != 0:
                if best_test['val_wt_f1_score'] < val_f1_weighted:
                    best_test['val_wt_f1_score'] = val_f1_weighted
                if best_test['val_accuracy'] < val_acc:
                    best_test['val_accuracy'] = val_acc

    # ----- FINAL EVALUATION STEP, USING FULLY TRAINED MODEL -----
    # print("--- Finished Training - Entering Final Evaluation Step\n")
    # # saving the model.
    # # /app/timeseries/multiclass_src
    # model_file_path = "/".join(["/app/timeseries/multiclass_src/models",
    #                             '{}_best_model_{}_{}_{}_{}.pth'.format(
    #                                 20201118, loss_metric, epoch, 0.5, run_name
    #                             )])
    # torch.save(model, model_file_path)
    # print("Saving best model to {}".format(model_file_path))

    # # inits.
    # model.eval()
    # test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

    # eval_json = {
    #     "run_name": None,
    #     "seed": seed,
    #     "0.1": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.2": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.3": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.4": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.45": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.5": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.55": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.6": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.7": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.8": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    #     "0.9": {"class_f1s": None, "mean_f1": None, "eval_dxn": None},
    # }

    # with torch.no_grad():
    #     for tau in test_thresholds:
    #         # go through all the thresholds, and test them out again.
    #         final_test_dxn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #         test_preds, test_labels = [], []
    #         for i, (inputs, labels) in enumerate(test_loader):
    #             # updating distribution of labels.
    #             labels_list = labels.numpy()
    #             for label in labels_list:
    #                 final_test_dxn[label] += 1

    #             # stacking onto tensors.
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             # passing it through our finalized model.
    #             output = model(inputs)
    #             labels = torch.zeros(len(labels), 10).to(device).scatter_(
    #                 1, labels.unsqueeze(1), 1.).to(device)

    #             pred_arr = output.detach().cpu().numpy()
    #             label_arr = labels.detach().cpu().numpy()

    #             # appending results.
    #             test_preds.append(pred_arr)
    #             test_labels.append(label_arr)

    #         test_preds = torch.tensor(test_preds[0])
    #         test_labels = torch.tensor(test_labels[0])

    #         class_f1s, mean_f1 = evaluation_f1(
    #             device=device, y_labels=test_labels, y_preds=test_preds, threshold=tau)

    #         tau = str(tau)
    #         eval_json[tau]['class_f1s'] = class_f1s.numpy().tolist()
    #         eval_json[tau]['mean_f1'] = mean_f1.item()
    #         eval_json[tau]['eval_dxn'] = final_test_dxn

    # eval_json['run'] = run_name
    # eval_json['seed'] = seed
    # record_results(eval_json, "results_thresh.json")

    # ----- recording results in a json.
    print(best_test)
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
    record_results(best_test, "results_1024.json")
    return


@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
@click.option("--imb", required=False, is_flag=True, default=False)
@click.option("--run_name", required=False)
@click.option("--cuda", required=False)
def run(loss, epochs, imb, run_name, cuda):
    # check if forcing imbalance
    print(run_name)
    print("Running on cuda: {}".format(cuda))
    imbalanced = False
    if imb:
        imbalanced = True

    # seeds = [1, 45, 92, 34, 15, 20, 150, 792, 3, 81]
    seeds = [1, 45, 92, 34, 15]
    # seeds = [20, 150, 792, 3, 81]
    # for i in range(len(seeds)):
    #     temp_name = str(run_name) + "-" + str(i + 5)
    #     train_cifar(loss_metric=loss, epochs=int(
    #         epochs), imbalanced=imbalanced, run_name=temp_name, seed=seeds[i], cuda=cuda)
    for i in range(len(seeds)):
        temp_name = str(run_name) + "-" + str(i)
        train_cifar(loss_metric=loss, epochs=int(
            epochs), imbalanced=imbalanced, run_name=temp_name, seed=seeds[i], cuda=cuda)


def main():
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = "C.UTF-8"
    run()


if __name__ == '__main__':
    main()

'''
python3 cifar.py --epochs=1000 --loss="approx-f1" --run_name="approx-f1-eval" --cuda=1

python3 cifar.py --epochs=1000 --loss="ce" --run_name="baseline-f1-imb-eval" --cuda=2


'''
