import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(VGGBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

    def forward(self, x):
        x = self.block(x)
        return x

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.block1 = VGGBlock(3, 64)
        self.block2 = VGGBlock(64, 128)
        self.block3 = VGGBlock(128, 256)
        self.block4 = VGGBlock(256, 512)
        self.block5 = VGGBlock(512, 512)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def loss(self, pred, label):
        return F.mse_loss(pred, label)
