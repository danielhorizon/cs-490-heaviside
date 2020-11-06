import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # TODO(dlee) - check if we're having any probabilities
        x = self.softmax(x)
        # print(x)
        return x


def run(): 
    model = Net()
    x = torch.randn(1, 32, 32, 3)

    # regular: input: torch.Size([1, 3, 32, 32])
    
    # input: torch.Size([1, 32, 32, 3]) -> imb 

    out = model(x) 

if __name__ == "__main__":
    run() 
