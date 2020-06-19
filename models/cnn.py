import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import *

class CNN(torch.nn.Module):
    def __init__(self, input_shape, features=[32, 32, 64, 64, 128, 256], split=1):
        super(CNN, self).__init__()
        self.name = "CNN"
        # self.fuse_in = nn.Conv2d(input_shape [0], features [0], 5, stride=1, padding=2)
        # self.fuse_in = FuseIn (input_shape [0], features[0] // 2, split=split)
        self.conv1 = Residual_Conv (input_shape [0], features[0], bias=True)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = Residual_Conv (features [0], features[1], bias=True)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = Residual_Conv (features [1], features[2], bias=True)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = Residual_Conv (features [2], features[3], bias=True)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.conv5 = Residual_Conv (features [3], features[4], bias=True)
        self.maxp5 = nn.MaxPool2d(2, 2)
        self.conv6 = Residual_Conv (features [4], features[5], bias=True)
        self.maxp6 = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        x = inputs
        # x = self.fuse_in (inputs)
        x = self.maxp1(self.conv1(x))
        x = self.maxp2(self.conv2(x))
        x = self.maxp3(self.conv3(x))
        x = self.maxp4(self.conv4(x))
        x = self.maxp5(self.conv5(x))
        x = self.maxp6(self.conv6(x))

        return x.view (x.size (0), -1)

class CNN7 (torch.nn.Module):
    def __init__(self, input_shape, features=[32, 32, 64, 64, 128, 256, 512], split=1):
        super(CNN7, self).__init__()
        self.name = "CNN7"
        # self.fuse_in = nn.Conv2d(input_shape [0], features [0], 5, stride=1, padding=2)
        # self.fuse_in = FuseIn (input_shape [0], features[0] // 2, split=split)
        self.conv1 = Residual_Conv (input_shape [0], features[0], bias=True)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = Residual_Conv (features [0], features[1], bias=True)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = Residual_Conv (features [1], features[2], bias=True)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = Residual_Conv (features [2], features[3], bias=True)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.conv5 = Residual_Conv (features [3], features[4], bias=True)
        self.maxp5 = nn.MaxPool2d(2, 2)
        self.conv6 = Residual_Conv (features [4], features[5], bias=True)
        self.maxp6 = nn.MaxPool2d(2, 2)
        self.conv7 = Residual_Conv(features[5], features[6], bias=True)
        self.maxp7 = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        x = inputs
        # x = self.fuse_in (inputs)
        x = self.maxp1(self.conv1(x))
        x = self.maxp2(self.conv2(x))
        x = self.maxp3(self.conv3(x))
        x = self.maxp4(self.conv4(x))
        x = self.maxp5(self.conv5(x))
        x = self.maxp6(self.conv6(x))
        x = self.maxp7(self.conv7(x))

        return x.view (x.size (0), -1)