import torch 
import numpy as np 
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from grace_torchconfusion import mean_f1_approx_loss_on

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)                # NEW ADDITIION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.softmax(x)                             # NEW ADDITION
        return x

def load_model(): 
    model = AlexNet().to("cuda:3")
    checkpoint = torch.load(
        '/app/timeseries/imagenet/trained-models/af1_checkpoint.pth.tar')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model 


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    acc1_arr, acc5_arr = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            print("Calculating batch...{}".format(i))
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            _, predicted = torch.max(output, 1)
            print(target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_arr.append(acc1.detach().item())
            acc5_arr.append(acc5.detach().item())

    return acc1_arr, acc5_arr


def evaluation_step(model): 
    valdir = os.path.join('/app/timeseries/imagenet/data', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    
    

