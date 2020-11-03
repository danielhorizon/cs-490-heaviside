import click
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics

# for early stopping.
from pytorchtools import EarlyStopping
from mc_torchconfusion import *

torch.manual_seed(0)
np.random.seed(0)

# displaying images:


def show_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_data(show=False, imbalanced=None):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # transform - transforms data during creation, downloads it locally, stores it in root, is train
    dataset = CIFAR10(train=True, download=True,
                      root="../data", transform=transform)
    test_data = CIFAR10(train=False, download=True,
                        root="../data", transform=transform)
    print("Train Size: {}".format(len(dataset)))
    print("Test Size: {}".format(len(test_data)))

    if imbalanced:
        # Get all training targets and count the number of class instances
        targets = np.array(dataset.labels)
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        print(class_counts)


    torch.manual_seed(1)
    val_size = 5000
    train_size = len(dataset) - val_size

    # Splitting into train/test/validation
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # forming batches, putting into loader:
    batch_size = 128
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    # loading the dataset --> DataLoader class (torch.utils.data.DataLoader)
    classes = dataset.classes
    print("Classes: {}".format(classes))

    # showing image
    if show:
        # getting random training images
        dataiter = iter(train_loader)
        images, labels = dataiter.next()

        # showing images
        show_image(torchvision.utils.make_grid(images))
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    return train_loader, val_loader, test_loader

# https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
@click.option("--imb", required=False, is_flag=True, default=False)
def run(loss, epochs, imb):
    # check if forcing imbalance
    imbalanced = False
    if imb:
        imbalanced = True

    # train
    # train_cifar(loss_metric=loss, epochs=int(epochs), imbalanced=imbalanced)
    load_data(show=False, imbalanced=imbalanced)


def main():
    run()


if __name__ == '__main__':
    main(),
