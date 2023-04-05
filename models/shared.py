import torch
import os, sys
from tqdm import tqdm
from data.process import get_iou, get_bounding_box
from torchvision import transforms

def train(model, device, train_loader, optimizer, epoch, validate=False, validation_loader=None):
    model.train()
    model = model.to(device)
    train_iou_track, train_l1_track = [], []
    val_iou_track, val_l1_track = [], []

    pbar = tqdm(total=epoch, desc='Training', position=0)
    for e in range(epoch):
        train_iou_score, train_l1_score = 0, 0
        val_iou_score, val_l1_score = None, None

        for _, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            prediction = model(image)
            loss = l1_loss(prediction, label)
            loss.backward()
            optimizer.step()

            train_iou_score += get_iou_score(prediction, label)
            train_l1_score += loss

        train_iou_score /= len(train_loader)
        train_l1_score /= len(train_loader)
        train_iou_track.append(train_iou_score)
        train_l1_track.append(train_l1_score.item())

        if validate:
            with torch.no_grad():
                val_iou_score, val_l1_score = 0, 0

                for _, (image, label) in enumerate(validation_loader):
                    image, label = image.to(device), label.to(device)
                    prediction = model(image)
                    val_iou_score += get_iou_score(prediction, label)
                    val_l1_score += l1_loss(prediction, label).item()

                val_iou_score /= len(validation_loader)
                val_l1_score /= len(validation_loader)
                val_iou_track.append(val_iou_score)
                val_l1_track.append(val_l1_score)
        
        pbar.update(1)
        if validate:
            pbar.write(f"Epoch: {e} train_iou: {train_iou_score:.5f} train_loss: {train_l1_score:.5f} val_iou: {val_iou_score:5f} val_loss: {val_l1_score:5f}")
        else:
            pbar.write(f"Epoch: {e} train_iou: {train_iou_score:.5f} train_loss: {train_l1_score:.5f}")
        pbar.refresh()

    return train_iou_track, train_l1_track, val_iou_track, val_l1_track
          
def test(model, device, test_loader):
    model.eval()
    model = model.to(device)

    test_iou_score, test_l1_score = 0, 0

    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            prediction = model(image)
            test_iou_score += get_iou_score(prediction, label)
            test_l1_score += l1_loss(prediction, label)

    test_iou_score /= len(test_loader)
    test_l1_score /= len(test_loader)
    sys.stdout.write(f"Test iou: {test_iou_score:.5f} loss: {test_l1_score:.5f}")

    return test_iou_score, test_l1_score

def predict_image(model, image):
    model.eval()
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((360, 360)),
            transforms.ToTensor()
        ])
    with torch.no_grad():
        image = transform(image)[None, :]
        prediction = model(image)
    return prediction[0]
            
def l1_loss(p, l):
    loss = torch.mean(torch.abs(p[:, 0] - l[:, 0]) + torch.abs(p[:, 1] - l[:, 1]) + torch.abs(p[:, 2] - l[:, 2]) + torch.abs(p[:, 3] - l[:, 3]))
    return loss

def get_iou_score(prediction, label):
    iou_score = 0
    for p, l in zip(prediction, label):
        iou_score += get_iou([(x.item(), y.item()) for x, y in get_bounding_box(*p)], [(x.item(), y.item()) for x, y in get_bounding_box(*l)])
    return iou_score / len(prediction)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

    



    
    
