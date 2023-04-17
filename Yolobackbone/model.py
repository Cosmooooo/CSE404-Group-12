import torch.nn as nn
import torch.nn.functional as F

class YoloBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(YoloBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3),
            nn.BatchNorm2d(outchannel),
            nn.ReLU())

    def forward(self, x):
        x = self.block(x)
        return x

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, 7),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(),
                    nn.Conv2d(64, 192, 3, 1, 1),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(),
                    nn.Conv2d(192, 128, 1, 1, 1),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, 1, 1),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU())
        self.block1 = YoloBlock(256, 512)
        self.block2 = YoloBlock(512, 1024)
        self.conv2 = nn.Sequential(
                    nn.Conv2d(1024, 1024, 3, 1, 1),
                    nn.ReLU())
        self.conv3 = nn.Sequential(
                    nn.Conv2d(1024, 1024, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, 3, 1, 1),
                    nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 4)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 4)
        x = self.conv3(x)
        x = F.avg_pool2d(x, 2)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def loss(self, pred, label):
        return F.mse_loss(pred, label)