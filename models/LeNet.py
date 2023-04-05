from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

"""
Simple LeNet Implementation
"""
class LeNet(Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2d(3,20,4)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(2,2)
        self.conv2 = Conv2d(20,50,2)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(2,2)
        self.fc1 = Linear(800,500)
        self.relu3 = ReLU()
        self.fc2 = Linear(500,100)
        self.logSoftmax = LogSoftmax(1)
        self.fc3 = Linear(100,4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        x = self.fc3(x)
        return x
