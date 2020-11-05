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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import metrics

# for early stopping.
from pytorchtools import EarlyStopping
from mc_torchconfusion import *

torch.manual_seed(0)
np.random.seed(0)


def show_image(loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
        print(fig)


def check_class_balance(dataset):
    targets = np.array(dataset.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)
    print(class_counts)


def create_imbalance(dataset):
    check_class_balance(dataset)
    targets = np.array(dataset.targets)
    # Create artificial imbalanced class counts
    # One of the classes has 805 of observations removed
    imbal_class_counts = [6000, 6000, 6000,
                          6000, 6000, 6000, 6000, 6000, 6000, 4500]

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


def load_data(show=False, imbalanced=None):
    torch.manual_seed(1)

    transform = transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])

    # train and test data 
    dataset = MNIST(root='../data', train=True, download=True,
                                         transform=transform)
    test_data = MNIST(root='../data', train=False, download=True,
                                           transform=transform)
    
    # Should roughly be a 10th of how big the train set is (60,000)
    val_size = 6000
    if imbalanced:
        dataset = create_imbalance(dataset)
        val_size = 5000
    
    print("Train Size: {}, Test Size: {}".format(len(dataset), len(test_data)))

    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # train and test loader 
    # forming batches, putting into loader:
    batch_size = 128
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # loading the dataset --> DataLoader class (torch.utils.data.DataLoader)
    classes = dataset.classes
    print("Classes: {}".format(classes))

    if show:
        show_image(test_loader)

    return train_loader, val_loader, test_loader


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


def check_freq(x):
    return np.array(np.unique(x, return_counts=True)).T


def train_mnist(loss_metric=None, epochs=None, imbalanced=None):
    using_gpu = False 
    if torch.cuda.is_available():
        print("device = cuda")
        device = "cuda"
        using_gpu = True
    else:
        print("device = cpu")
        device = "cpu"
    print("using DEVICE: {}".format(device))

    # loading in data 
    train_loader, val_loader, test_loader = load_data(show=False, imbalanced=imbalanced)
    
    first_run = True 
    approx = False 
    model = Net().to(device)

    train_losses = []
    train_counter = []
    test_losses = []

    log_interval = 10
    torch.manual_seed(1)

    model = Net().to(device)
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    best_test = {
        "best-epoch": 0,
        "loss": float('inf'),
        "f1_score": 0,
        "accuracy": 0
    }

    # ----- TRAINING -----
    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        accs, microf1s, macrof1s, wf1s = [], [], [], []
        # going over in batches of 128
        for i, (inputs, labels) in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(labels[0])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs)  # batchsize * 10
            # print(output[0]) # this looks fine, we have 10 values in here.

            if not approx:
                loss = criterion(output, labels)
            else:
                train_labels = torch.zeros(len(labels), 10).to(device).scatter_(
                    1, labels.unsqueeze(1), 1.).to(device)

                loss = criterion(y_labels=train_labels, y_preds=output)

            losses.append(loss)
            loss.backward()
            optimizer.step()

            # print statistics; every 2000 mini-batches
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            ## check prediction
            model.eval()
            y_pred = model(inputs)
            _, train_preds = torch.max(y_pred, 1)

            accs.append(accuracy_score(
                y_true=labels.cpu(), y_pred=train_preds.cpu()))
            microf1s.append(f1_score(y_true=labels.cpu(),
                                     y_pred=train_preds.cpu(), average="micro"))
            macrof1s.append(f1_score(y_true=labels.cpu(),
                                     y_pred=train_preds.cpu(), average="macro"))
            wf1s.append(f1_score(y_true=labels.cpu(),
                                 y_pred=train_preds.cpu(), average="weighted"))

        print("Train - Epoch ({}): | Acc: {:.4f} | W F1: {:.4f} | Micro F1: {:.4f}| Macro F1: {:.4f}".format(
            epoch, np.array(accs).mean(), np.array(microf1s).mean(),
            np.array(macrof1s).mean(), np.array(
                microf1s).mean(), np.array(wf1s).mean()
        )
        )

        if using_gpu:
            mloss = torch.mean(torch.stack(losses))
        else:
            mloss = np.array([x.item for x in losses]).mean()
        # ----- TEST SET -----
        # Calculate metrics after going through all the batches
        model.eval()
        test_preds, test_labels = np.array([]), np.array([])
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            # print(output[0])

            _, predicted = torch.max(output, 1)
            # print(predicted[0])

            pred_arr = predicted.cpu().numpy()
            # print("test:{}".format(list(set(pred_arr))))
            label_arr = labels.cpu().numpy()

            test_labels = np.concatenate([test_labels, label_arr])
            test_preds = np.concatenate([test_preds, pred_arr])

        test_acc = accuracy_score(y_true=test_labels, y_pred=test_preds)
        test_f1_micro = f1_score(
            y_true=test_labels, y_pred=test_preds, average='micro')
        test_f1_macro = f1_score(
            y_true=test_labels, y_pred=test_preds, average='macro')
        test_f1_weighted = f1_score(
            y_true=test_labels, y_pred=test_preds, average='weighted')

        if best_test['loss'] > mloss:
            best_test['loss'] = mloss
            best_test['best-epoch'] = epoch
        if best_test['f1_score'] < test_f1_weighted:
            best_test['f1_score'] = test_f1_weighted
        if best_test['accuracy'] < test_acc:
            best_test['accuracy'] = test_acc

        print("Test - Epoch ({}): | Acc: {:.4f} | W F1: {:.4f} | Micro F1: {:.4f} | Macro F1: {:.4f}".format(
            epoch, test_acc, test_f1_weighted, test_f1_micro, test_f1_macro)
        )
        print(list(set(test_preds)))
        print("Count of 9's in Preds: {} and Labels: {}".format(
            np.count_nonzero(test_preds == 9.0), np.count_nonzero(test_labels == 9.0)))

        # 0 = airplane, 1 = automobile, 2 = bird, 3 = cat, 4 = deer, 5 = dog, 6 = frog, 7 = horse, 8 = ship, 9 = truck
        print(classification_report(y_true=test_labels, y_pred=test_preds,
                                    target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

        # ----- VALIDATION SET -----
        # Calculate metrics after going through all the batches
        model.eval()
        valid_losses = []
        with torch.no_grad():
            val_preds, val_labels = np.array([]), np.array([])
            for i, (inputs, labels) in enumerate(val_loader):
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
                        1, labels.unsqueeze(1), 1.)
                    curr_val_loss = criterion(
                        y_labels=valid_labels, y_preds=output)
                else:
                    curr_val_loss = criterion(output, labels)

                valid_losses.append(curr_val_loss.detach().cpu().numpy())

            val_acc = accuracy_score(y_true=val_labels, y_pred=val_preds)
            val_f1_micro = f1_score(
                y_true=val_labels, y_pred=val_preds, average='micro')
            val_f1_macro = f1_score(
                y_true=val_labels, y_pred=val_preds, average='macro')
            val_f1_weighted = f1_score(
                y_true=val_labels, y_pred=val_preds, average='weighted')

            valid_loss = np.mean(valid_losses)
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

            print("Val - Epoch ({}): | Acc: {:.4f} | W F1: {:.4f} | Micro F1: {:.4f} | Macro F1: {:.4f}\n".format(
                epoch, val_acc, val_f1_weighted, val_f1_micro, val_f1_macro)
            )

    print(best_test)
    return


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
