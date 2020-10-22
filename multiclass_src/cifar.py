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
    train_loader = DataLoader(train_ds, batch_size=4,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=4, num_workers=4)

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
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
    model = Net()

    train_loader, _, test_loader = load_data(show=False)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # TODO(dlee): add the metrics in when training. 
    if loss_metric == "ce": 
        criterion = nn.CrossEntropyLoss()
    else: 
        criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # going over in batches of 4 
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics; every 2000 mini-batches 
            running_loss += loss.item()
            if i % 2000 == 1999:    
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('--Finished Training')

    # Test
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
    
@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
def run(loss, epochs):
    train_cifar(loss_metric=loss, epochs=int(epochs))

def main():
    run(3)

if __name__ == '__main__':
    main()



    





    
