import os, sys
sys.path.append("/home/cosmo/Desktop/cse404/")

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from common.utils import *
from Dataset import *
from common.process import get_bounding_box, get_iou, estimate_iou


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ROIPooling(nn.Module):
    def __init__(self, output_size):
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.roi_pooling = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, features, rois):
        result = []
        for i in range(rois.shape[0]):
            x = []
            for f, r in zip(features, rois):
                # f = get_region(f, r)
                f = f
                f = self.roi_pooling(f)
                x.append(f)
            x = torch.cat(x, dim=0)
        result.append(x)
        result = torch.stack(result).view(features.shape[0], -1, *self.output_size)
        return result
    
class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes

        base_model = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.features.children())[:-1])
        self.roi_pooling = ROIPooling((7, 7))
        self.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])

        self.cls = nn.Linear(4096, self.num_classes)
        self.reg = nn.Linear(4096, 4 * self.num_classes)

    def forward(self, image, rois):
        feature = self.feature_extractor(image)
        feature = self.roi_pooling(feature, rois)
        feature = feature.view(feature.size(0), -1)
        feature = self.classifier(feature)
        cls = self.cls(feature)
        reg = self.reg(feature).view(-1, self.num_classes, 4)
        return cls, reg
    
    def loss(self, pred, label):
        cls, reg = pred
        cls_label, reg_label = label

        cls_loss = nn.CrossEntropyLoss()(cls, cls_label)
        reg_loss = nn.SmoothL1Loss()(reg[torch.arange(reg.size(0)), cls_label], reg_label[torch.arange(reg.size(0)), cls_label])
        return cls_loss + reg_loss

def train(dataloader, model, optimizer, lr_scheduler):
    model.train()

    train_loss = 0.0

    for image, rois, cls_label, reg_label in dataloader:
        image = image.to(device)
        rois = rois.to(device)
        cls_label = cls_label.to(device)
        reg_label = reg_label.to(device)

        optimizer.zero_grad()

        cls, reg = model(image, rois)

        loss = fastrcnn.loss((cls, reg), (cls_label, reg_label))
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item() * image.size(0)

    train_loss /= dataloader.dataset.__len__()
    return train_loss

def test(dataloader, model):
    model.eval()

    test_loss = 0.0

    with torch.no_grad():
        for image, rois, cls_label, reg_label in dataloader:
            image = image.to(device)
            rois = rois.to(device)
            cls_label = cls_label.to(device)
            reg_label = reg_label.to(device)

            cls, reg = model(image, rois)

            loss = fastrcnn.loss((cls, reg), (cls_label, reg_label))

            test_loss += loss.item() * image.size(0)

    test_loss /= dataloader.dataset.__len__()
    return test_loss

def predict(image, model, rois):
    model.eval()

    image = image.to(device)

    cls, reg = model(image, rois)
    return cls, reg


fastrcnn = FastRCNN(2).to(device)
optimizer = optim.Adam(fastrcnn.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

dataset = FastRCNNDataset("/home/cosmo/Desktop/cse404/data.json")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

train_loss = []
test_loss = []


# for epoch in range(10):
#     train_loss.append(train(dataloader, fastrcnn, optimizer, lr_scheduler))
#     test_loss.append(test(dataloader, fastrcnn))
    
#     print("Epoch: {}, Train Loss: {}, Test Loss: {}".format(epoch, train_loss[-1], test_loss[-1]))

# # plt.plot(train_loss, label="Train Loss")
# # plt.plot(test_loss, label="Test Loss")
# # plt.legend()
# # plt.show()
# image = dataset[0][0].view(-1, 3, 224, 224)
# rois = dataset[0][1]
# cls, reg = predict(image, fastrcnn, rois)
# print(cls)
# print(reg)

    
