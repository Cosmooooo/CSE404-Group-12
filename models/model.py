import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 30 * 30, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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
        for _, (frames, label) in enumerate(test_loader):
            starting_frame = frames[0]
            starting_frame, label = starting_frame.to(device), label.to(device)
            prediction = model(starting_frame)
            loss = custom_loss(prediction, label)
            loss_track.append(loss)
            total_loss += loss
    return loss_track, total_loss
            
def custom_loss(prediction, label):
    p_left, p_top, p_scale, p_rot = prediction[:]
    l_left, l_top, l_scale, l_rot = label[0,:]
    loss = torch.mean(torch.abs(p_left - l_left) + torch.abs(p_top - l_top) + torch.abs(p_scale - l_scale) + torch.abs(p_rot - l_rot))
    return loss

def save_model(model, path):
    root = 'models/checkpoints/'
    torch.save(model.state_dict(), root+path)

def load_model(model, path):
    root = 'models/checkpoints/'
    model.load_state_dict(torch.load(root+path))
    return model

    



    
    
