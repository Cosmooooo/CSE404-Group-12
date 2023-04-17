import torch
import torch.nn as nn

import torchvision.models as models

class ROIPooling(nn.Module):
    def __init__(self, output_size):
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.roi_pooling = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, features_batch, rois_batch):
        result = []
        feature_w = features_batch.size(3)
        feature_h = features_batch.size(2)

        x1 = torch.floor(rois_batch[:, :, 0]*feature_w).type(torch.int32)
        y1 = torch.floor(rois_batch[:, :, 1]*feature_h).type(torch.int32)
        x2 = torch.ceil(rois_batch[:, :, 2]*feature_w).type(torch.int32)
        y2 = torch.ceil(rois_batch[:, :, 3]*feature_h).type(torch.int32)

        for i, (feature, rois) in enumerate(zip(features_batch, rois_batch)):
            feature = feature.unsqueeze(0)
            for j in range(rois.size(0)):
                roi = feature[:, :, y1[i][j]:y2[i][j], x1[i][j]:x2[i][j]]
                roi = self.roi_pooling(roi)
                result.append(roi)
        result = torch.cat(result, dim=0)
        return result
    
class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes

        base_model = models.vgg11() # weights=VGG16_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.features.children())[:-1])
        self.roi_pooling = ROIPooling((7, 7))
        self.base_classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])

        self.classifier = nn.Linear(4096, self.num_classes)
        self.regressor = nn.Linear(4096, self.num_classes * 4)

    def forward(self, images_batch, rois_batch):
        feature_batch = self.feature_extractor(images_batch)
        feature_batch = self.roi_pooling(feature_batch, rois_batch)
        feature_batch = feature_batch.view(feature_batch.size(0), -1)
        feature_batch = self.base_classifier(feature_batch)

        classes_pred = self.classifier(feature_batch)
        bboxes_pred = self.regressor(feature_batch).view(-1, self.num_classes, 4)

        return classes_pred, bboxes_pred
    
    def loss(self, pred, label, k=1):
        classes_pred, bboxes_pred = pred
        classes_label, bboxes_label = label

        classes_label = classes_label.reshape(-1)

        cls_loss = nn.CrossEntropyLoss()(classes_pred, classes_label)

        lbl = classes_label.view(-1, 1, 1).expand(-1, 1, 4)
        mask = (classes_label != 0).float().unsqueeze(1).expand(-1, 4)
        reg_loss = nn.SmoothL1Loss()(bboxes_pred.gather(1, lbl).squeeze(1) * mask, bboxes_label.view(-1, 4) * mask) * k
        return cls_loss + reg_loss, cls_loss, reg_loss

