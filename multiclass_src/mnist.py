import click
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import metrics

# torchconfuson files
from mc_torchconfusion import *

def check_class_balance(dataset):
    targets = np.array(dataset.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)
    print(class_counts)

def create_imbalance(dataset):
    check_class_balance(dataset)
    targets = np.array(dataset.targets)
    # Create artificial imbalanced class counts
    imbal_class_counts = [6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 4500]

    # Get class indices
    class_indices = [np.where(targets == i)[0] for i in range(10)]

    # Get imbalanced number of instances
    imbal_class_indices = [class_idx[:class_count] for class_idx,
                           class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)
    print("imbalanced class indices: {}".format(imbal_class_indices))

    # Set target and data to dataset
    dataset.targets = targets[imbal_class_indices]
    dataset.data = dataset.data[imbal_class_indices]

    assert len(dataset.targets) == len(dataset.data)
    print("After imbalance: {}".format(check_class_balance(dataset)))

    return dataset


def load_balanced_data(show=False, shuffle=True, seed=None, batch_size=None):
    transform = transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])

    # train and test data 
    train_dataset = MNIST(root='../data', train=True, download=True, transform=transform)
    valid_dataset = MNIST(root='../data', train=True, download=True, transform=transform)

    num_train = len(train_dataset)
    valid_size = 0.2
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle == True:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,num_workers=0)

    ## getting testing data
    test_dataset = MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
    return train_loader, valid_loader, test_loader


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
    train_mnist(loss_metric=loss, epochs=int(epochs), imbalanced=imbalanced)


def main():
    run()


if __name__ == '__main__':
    main(),
