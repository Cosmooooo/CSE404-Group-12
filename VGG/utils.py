import sys
sys.path.append("/home/cosmo/Desktop/cse404/")

from tqdm import tqdm
import torch
import numpy as np
import cv2

from VGG.model import VGG
from common.utils import plot, calculate_iou
from common.processVideo import get_starting_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, optimizer, dataloader, e):
    model.train()
    
    epoch_loss, epoch_iou = 0, 0
    
    for images, bbox_labels, _,_,_ in tqdm(dataloader, desc=f"Training Epoch {e}", leave=False):
        images = images.to(device)
        bbox_labels = bbox_labels.to(device)
        
        optimizer.zero_grad()

        pred = model(images).view(len(images), 1, 4)
        loss = model.loss(pred, bbox_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        bbox_labels = bbox_labels.cpu().detach().numpy()
        bboxes_pred = pred.cpu().detach().numpy()

        for i in range(len(images)):
            iou = calculate_iou(bboxes_pred[i], bbox_labels[i])
            epoch_iou += sum(iou)
    
    return epoch_loss / len(dataloader), epoch_iou / len(dataloader.dataset)

def test_epoch(model, dataloader, e):
    model.eval()
    
    epoch_loss, epoch_iou = 0, 0
    
    with torch.no_grad():
        for images, bbox_labels, _,_,_ in tqdm(dataloader, desc=f"Validating Epoch {e}", leave=False):
            images = images.to(device)
            bbox_labels = bbox_labels.to(device)
            
            pred = model(images).view(len(images), 1, 4)

            loss = model.loss(pred, bbox_labels)

            epoch_loss += loss.item()

            bbox_labels = bbox_labels.cpu().detach().numpy()
            bboxes_pred = pred.cpu().detach().numpy()

            for i in range(len(images)):
                iou = calculate_iou(bboxes_pred[i], bbox_labels[i])
                epoch_iou += sum(iou)
        
    return epoch_loss / len(dataloader), epoch_iou / len(dataloader.dataset)

def train(train_loader, validation_loader, optimizer, epoch, lr=1e-3):
    model = VGG().to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    train_losses, train_ious = [], []
    val_losses, val_ious = [], []

    pbar = tqdm(range(epoch), desc="Training VGG")
    for e in range(epoch):
        val_loss, val_iou = test_epoch(model, validation_loader, e)
        train_loss, train_iou = train_epoch(model, optimizer, train_loader, e)
        
        train_losses.append(train_loss)
        train_ious.append(train_iou)

        val_losses.append(val_loss)
        val_ious.append(val_iou)

        pbar.write(f"Train Epoch: {e}, loss: {train_loss:.5f} IOU: {train_iou:.5f}")
        pbar.write(f"Validation Epoch: {e}, loss: {val_loss:.5f} IOU: {val_iou:.5f}")
        pbar.update(1)
    
    plot(title="VGG Train", sub_titles=["Loss", "IOU"], \
         train_result=[train_losses, train_ious], validate_result=[val_losses, val_ious])
    
    return model

def test(model, test_loader):
    test_loss, test_iou = test_epoch(model, test_loader, "-")
    print(f"Test Loss: {test_loss}, Test IOU: {test_iou}")

def predict_image(model, orig_image):
    h, w = orig_image.shape[:2]

    image = cv2.resize(orig_image, (224, 224))
    image = image.transpose(2, 0, 1)

    image = torch.from_numpy(image[np.newaxis, :]).float().to(device)

    bbox = model(image).cpu().detach().numpy()

    bbox[:, 0] *= w
    bbox[:, 1] *= h
    bbox[:, 2] *= w
    bbox[:, 3] *= h

    return orig_image, bbox.astype(np.int32)
    
def load(ckpt):
    model = VGG().to(device)
    model.load_state_dict(torch.load(ckpt)) 
    return model

if __name__ == "__main__":
    model = VGG().to(device)
    model.load_state_dict(torch.load("checkpoints/VGG.pth"))   
    image = get_starting_frame("/media/cosmo/Dataset/YTCelebrity/ytcelebrity/0115_03_023_al_gore.avi")
    image, bboxes = predict_image(model, image)
    for bbox in bboxes:
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.imwrite("VGG/example.jpg", image)
    cv2.waitKey(0)