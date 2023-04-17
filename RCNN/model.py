import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as T
    
class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        self.num_classes = num_classes
        self.transform = T.Resize((224, 224))

        base_model = models.alexnet() # weights=AlexNet_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
        self.base_classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
        self.classifier = nn.Linear(4096, self.num_classes)
        self.regressor = nn.Linear(4096, self.num_classes * 4)

    def forward(self, images_batch, rois_batch):
        feature_batch = []
        image_w = images_batch.size(3)
        image_h = images_batch.size(2)

        x1 = torch.floor(rois_batch[:, :, 0]*image_w).type(torch.int32)
        y1 = torch.floor(rois_batch[:, :, 1]*image_h).type(torch.int32)
        x2 = torch.ceil(rois_batch[:, :, 2]*image_w).type(torch.int32)
        y2 = torch.ceil(rois_batch[:, :, 3]*image_h).type(torch.int32)

        for i, (image, rois) in enumerate(zip(images_batch, rois_batch)):
            for j in range(rois.size(0)):
                roi = image[:, y1[i][j]:y2[i][j], x1[i][j]:x2[i][j]]
                roi = self.transform(roi)
                feature_batch.append(roi)
        feature_batch = torch.stack(feature_batch, dim=0)

        feature_batch = self.feature_extractor(feature_batch)
        feature_batch = feature_batch.view(feature_batch.size(0), -1)
        feature_batch = self.base_classifier(feature_batch)

        classes_pred = self.classifier(feature_batch)
        bboxes_pred = self.regressor(feature_batch).view(-1, self.num_classes, 4)

        return classes_pred, bboxes_pred
    
    def cls_loss(self, pred, label):
        classes_pred, _ = pred
        classes_label, _ = label

        classes_label = classes_label.reshape(-1)

        cls_loss = nn.CrossEntropyLoss()(classes_pred, classes_label)
        return cls_loss
    
    def reg_loss(self, pred, label, k=1):
        _, bboxes_pred = pred
        classes_label, bboxes_label = label

        classes_label = classes_label.reshape(-1)

        lbl = classes_label.view(-1, 1, 1).expand(-1, 1, 4)
        mask = (classes_label != 0).float().unsqueeze(1).expand(-1, 4)
        reg_loss = nn.SmoothL1Loss()(bboxes_pred.gather(1, lbl).squeeze(1) * mask, bboxes_label.view(-1, 4) * mask) * k
        return reg_loss
