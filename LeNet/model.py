import torch.nn as nn
import torch.nn.functional as F

"""
Simple LeNet Implementation
"""
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3,20,4)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(20,50,2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(50 * 54 * 54, 500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500,4)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def loss(self, pred, label):
        return F.mse_loss(pred, label)