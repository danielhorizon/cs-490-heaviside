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


def load_data(show=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # transform - transforms data during creation, downloads it locally, stores it in root, is train 
    dataset = CIFAR10(train=True, download=True, root="../data", transform=transform)
    test_data = CIFAR10(train=False, download=True, root="../data", transform=transform)
    print("Train Size: {}".format(len(dataset)))
    print("Test Size: {}".format(len(test_data)))

    torch.manual_seed(1)
    val_size = 5000
    train_size = len(dataset) - val_size

    # Splitting into train/test/vallidation
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # forming batches, putting into loader:
    batch_size=128 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
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

'''
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_cifar(loss_metric=None, epochs=None): 
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
    train_loader, val_loader, test_loader = load_data(show=False)

    # setting inits, initialize the early_stopping object
    first_run = True 
    approx = False
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
    model = Net().to(device)
    patience = 100
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
        accs =  []
        microf1s, macrof1s, wf1s = [], [], [] 
        # going over in batches of 128
        for i, (inputs, labels) in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs) # batchsize * 10 
            
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

            accs.append(accuracy_score(y_true=labels.cpu(), y_pred=train_preds.cpu()))
            microf1s.append(f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="micro"))
            macrof1s.append(f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="macro"))
            wf1s.append(f1_score(y_true=labels.cpu(), y_pred=train_preds.cpu(), average="weighted"))
        
        print("Train - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Macro F1: {:.3f}".format(
                epoch, np.array(accs).mean(), np.array(microf1s).mean(), 
                np.array(macrof1s).mean(), np.array(wf1s).mean()
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
            _, predicted = torch.max(output, 1)

            pred_arr = predicted.cpu().numpy()
            label_arr = labels.cpu().numpy() 

            test_labels = np.concatenate([test_labels, label_arr])
            test_preds = np.concatenate([test_preds, pred_arr])

        test_acc = accuracy_score(y_true=test_labels, y_pred=test_preds)
        test_f1_micro = f1_score(y_true=test_labels, y_pred=test_preds, average='micro')
        test_f1_macro = f1_score(y_true=test_labels, y_pred=test_preds, average='macro')
        test_f1_weighted = f1_score(y_true=test_labels, y_pred=test_preds, average='weighted')

        if best_test['loss'] > mloss:
            best_test['loss'] = mloss
            best_test['best-epoch'] = epoch
        if best_test['f1_score'] < test_f1_weighted:
            best_test['f1_score'] = test_f1_weighted
        if best_test['accuracy'] < test_acc:
            best_test['accuracy'] = test_acc

        print("Test - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Macro F1: {:.3f}".format(
            epoch, test_acc, test_f1_weighted, test_f1_macro)
        )


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
            val_f1_micro = f1_score(y_true=val_labels, y_pred=val_preds, average='micro')
            val_f1_macro = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')
            val_f1_weighted = f1_score(y_true=val_labels, y_pred=val_preds, average='weighted')

            valid_loss = np.mean(valid_losses)
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

            print("Val - Epoch ({}): | Acc: {:.3f} | W F1: {:.3f} | Macro F1: {:.3f}".format(
                epoch, val_acc, val_f1_weighted, val_f1_macro)
            )

    print(best_test)
    return 

        
    
@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
def run(loss, epochs):
    train_cifar(loss_metric=loss, epochs=int(epochs))

def main():
    run()

if __name__ == '__main__':
    main()



    





    
