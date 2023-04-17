import sys
sys.path.append("/home/cosmo/Desktop/cse404/")

from tqdm import tqdm
import torch
import numpy as np
import cv2

from common.utils import plot, calculate_iou, transform_to, selective_search, scale_bbox
from common.processVideo import get_starting_frame
from FastRCNN.model import FastRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, optimizer, dataloader, e):
    model.train()
    
    epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_iou = 0, 0, 0, 0
    
    for images, bbox_labels, rois, rois_cls, rois_transform in tqdm(dataloader, desc=f"Training Epoch {e}", leave=False):
        images = images.to(device)
        bbox_labels = bbox_labels.to(device)
        rois = rois.to(device)
        rois_cls = rois_cls.to(device)
        rois_transform = rois_transform.to(device)
        
        optimizer.zero_grad()

        classes_pred, bboxes_pred = model(images, rois)
        loss, cls_loss, reg_loss = model.loss((classes_pred, bboxes_pred), (rois_cls, rois_transform), k=5)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_reg_loss += reg_loss.item()

        rois = rois.cpu().detach().numpy()
        bbox_labels = bbox_labels.cpu().detach().numpy()
        classes_pred = classes_pred.softmax(dim=1).view(len(images), -1, model.num_classes).cpu().detach().numpy()
        bboxes_pred = bboxes_pred.view(len(images), -1, model.num_classes, 4).cpu().detach().numpy()

        for i in range(len(images)):
            for j in range(1, model.num_classes): # ignore background
                class_transforms = bboxes_pred[i, :, j, :]
                class_rois = rois[i, :]

                class_bboxes = transform_to(class_rois, class_transforms)
                class_scores = classes_pred[i, :, j]

                index = np.argmax(class_scores)
                best_bbox = class_bboxes[index][np.newaxis, :]
               
                iou = calculate_iou(best_bbox, bbox_labels[i])
                epoch_iou += sum(iou)
    
    return epoch_loss / len(dataloader), epoch_cls_loss / len(dataloader),\
          epoch_reg_loss / len(dataloader), epoch_iou / len(dataloader.dataset) / (model.num_classes - 1)

def test_epoch(model, dataloader, e):
    model.eval()
    
    epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_iou = 0, 0, 0, 0

    with torch.no_grad():
        for images, bbox_labels, rois, rois_cls, rois_transform in tqdm(dataloader, desc=f"Validating Epoch {e}", leave=False):
            images = images.to(device)
            bbox_labels = bbox_labels.to(device)
            rois = rois.to(device)
            rois_cls = rois_cls.to(device)
            rois_transform = rois_transform.to(device)

            classes_pred, bboxes_pred = model(images, rois)
            loss, cls_loss, reg_loss = model.loss((classes_pred, bboxes_pred), (rois_cls, rois_transform), k=5)

            rois = rois.cpu().detach().numpy()
            bbox_labels = bbox_labels.cpu().detach().numpy()
            classes_pred = classes_pred.view(len(images), -1, model.num_classes).cpu().detach().numpy()
            bboxes_pred = bboxes_pred.view(len(images), -1, model.num_classes, 4).cpu().detach().numpy()

            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()

            for i in range(len(images)):
                for j in range(1, model.num_classes): # ignore background
                    class_transforms = bboxes_pred[i, :, j, :]
                    class_rois = rois[i, :]

                    class_bboxes = transform_to(class_rois, class_transforms)
                    class_scores = classes_pred[i, :, j]

                    index = np.argmax(class_scores)
                    best_bbox = class_bboxes[index][np.newaxis, :]
                
                    iou = calculate_iou(best_bbox, bbox_labels[i])
                    epoch_iou += sum(iou)
    
    return epoch_loss / len(dataloader), epoch_cls_loss / len(dataloader),\
          epoch_reg_loss / len(dataloader), epoch_iou / len(dataloader.dataset) / (model.num_classes - 1)

def train(train_loader, validation_loader, optimizer, epoch, lr=1e-3):
    model = FastRCNN(2).to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    train_losses, train_cls_losses, train_reg_losses, train_ious = [], [], [], []
    val_losses, val_cls_losses, val_reg_losses, val_ious = [], [], [], []

    pbar = tqdm(range(epoch), desc="Training Fast-RCNN")
    for e in range(epoch):
        val_loss, val_cls_loss, val_reg_loss, val_iou = test_epoch(model, validation_loader, e)
        train_loss, train_cls_loss, train_reg_loss, train_iou = train_epoch(model, optimizer, train_loader, e)
        
        train_losses.append(train_loss)
        train_cls_losses.append(train_cls_loss)
        train_reg_losses.append(train_reg_loss)
        train_ious.append(train_iou)

        val_losses.append(val_loss)
        val_cls_losses.append(val_cls_loss)
        val_reg_losses.append(val_reg_loss)
        val_ious.append(val_iou)

        pbar.write(f"Train Epoch: {e}, total loss: {train_loss:.5f} classifier loss: {train_cls_loss:.5f} regressor loss: {train_reg_loss:.5f} IOU: {train_iou:.5f}")
        pbar.write(f"Validation Epoch: {e}, total loss: {val_loss:.5f} classifier loss: {val_cls_loss:.5f} regressor loss: {val_reg_loss:.5f} IOU: {val_iou:.5f}")
        pbar.update(1)
    
    plot(title="Fast-RCNN Train", sub_titles=["Total Loss", "Classifier Loss", "Regressor Loss", "IOU"], \
         train_result=[train_losses, train_cls_losses, train_reg_losses, train_ious], validate_result=[val_losses, val_cls_losses, val_reg_losses, val_ious])
    
    return model

def test(model, test_loader):
    test_loss, test_cls_loss, test_reg_loss, test_iou = test_epoch(model, test_loader, "-")
    print(f"Test Loss: {test_loss}, Test Cls Loss: {test_cls_loss}, Test Reg Loss: {test_reg_loss} Test IOU: {test_iou}")


def predict_image(model, orig_image):
    image_size = orig_image.shape[:2]
    h, w = image_size

    roi = selective_search(orig_image)
    roi = scale_bbox(image_size, roi)

    image = cv2.resize(orig_image, (224, 224))
    image = image.transpose(2, 0, 1)

    image = torch.from_numpy(image[np.newaxis, :]).float().to(device)
    roi = torch.from_numpy(roi[np.newaxis, :]).float().to(device)

    classes_pred, bboxes_pred = model(image, roi)

    roi = roi.cpu().detach().numpy()
    classes_pred = classes_pred.view(len(image), -1, model.num_classes).cpu().detach().numpy()
    bboxes_pred = bboxes_pred.view(len(image), -1, model.num_classes, 4).cpu().detach().numpy()

    bboxes = []
    for i in range(1, model.num_classes): # ignore background
        class_transforms = bboxes_pred[0, :, i, :]
        class_rois = roi[0, :]

        class_bboxes = transform_to(class_rois, class_transforms)
        class_scores = classes_pred[0, :, i]

        index = np.argmax(class_scores)
        best_bbox = class_bboxes[index][np.newaxis, :]

        best_bbox[:, 0] *= w
        best_bbox[:, 1] *= h
        best_bbox[:, 2] *= w
        best_bbox[:, 3] *= h

        bboxes.append(best_bbox)
    
    bboxes = np.stack(bboxes, axis=0).squeeze(0)

    return orig_image, bboxes.astype(np.int32)

def load(ckpt):
    model = FastRCNN(2).to(device)
    model.load_state_dict(torch.load(ckpt))   
    return model
    
if __name__ == "__main__":
    model = FastRCNN(2).to(device)
    model.load_state_dict(torch.load("checkpoints/FastRCNN.pth"))   
    image = get_starting_frame("/media/cosmo/Dataset/YTCelebrity/ytcelebrity/0115_03_023_al_gore.avi")
    image, bboxes = predict_image(model, image)
    for bbox in bboxes:
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.imwrite("FastRCNN/example.jpg", image)
    cv2.waitKey(0)
