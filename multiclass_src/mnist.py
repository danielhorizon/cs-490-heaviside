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


def load_data(show=False, ibalanced=None):
    torch.manual_seed(1)
    batch_size = 128 

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
    print("Train Size: {}, Test Size: {}".format(
        train_loader.shape, test_loader.shape))

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


def train_mnist(loss_metric=None, epochs=None):
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
    
    first_run = True 
    approx = False 
    model = Net().to(device)

    

    train_losses = []
    train_counter = []
    test_losses = []

    log_interval = 10
    torch.manual_seed(1)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    if loss_metric == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # TRAIN MODEL
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

                # saving model
                # torch.save(model.state_dict(), '/results/model.pth')
                # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

        # TEST MODEL
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target,
                                        size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


@click.command()
@click.option("--loss", required=True)
@click.option("--epochs", required=True)
def run(loss, epochs):
    train_mnist(loss_metric='ce', epochs=int(epochs))


def main():
    run()


if __name__ == '__main__':
    main()
