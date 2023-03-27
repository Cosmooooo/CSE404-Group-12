import torch
import os

def train(model, device, train_loader, optimizer):
    model.train()
    model = model.to(device)
    loss_track = []
    total_loss = 0
    for _, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        prediction = model(image)
        loss = custom_loss(prediction, label)
        loss.backward()
        optimizer.step()
        loss_track.append(loss)
        total_loss += loss
    return loss_track, total_loss
          
def test(model, device, test_loader):
    model.eval()
    model = model.to(device)
    loss_track = []
    total_loss = 0
    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            prediction = model(image)
            loss = custom_loss(prediction, label)
            loss_track.append(loss)
            total_loss += loss
    return loss_track, total_loss
            
def custom_loss(prediction, label):
    loss = 0
    for p, l in zip(prediction, label):
        p_left, p_top, p_scale, p_rot = p
        l_left, l_top, l_scale, l_rot = l
        loss += torch.mean(torch.abs(p_left - l_left) + torch.abs(p_top - l_top) + torch.abs(p_scale - l_scale) + torch.abs(p_rot - l_rot))
    return loss

def save_model(model, path):
    root = 'checkpoints/'
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(model.state_dict(), root+path)

def load_model(model, path):
    root = 'checkpoints/'
    model.load_state_dict(torch.load(root+path))
    return model

    



    
    
