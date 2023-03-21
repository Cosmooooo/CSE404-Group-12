import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.YTCelebrityDataset import YTCelebrityDataset
from models.model import Model, train, test
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
epoch = 10
root = "/media/cosmo/Dataset/YTCelebrity/ytcelebrity/"
csv_path = "celebrity.csv"

def main():
    dataset = YTCelebrityDataset(root, csv_path)

    generator = torch.Generator().manual_seed(42)
    training, testing = torch.utils.data.random_split(dataset, [0.7, 0.3], generator=generator)

    train_loader = DataLoader(training, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(testing, batch_size=4, shuffle=True, num_workers=4)

    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train(model, device, train_loader, optimizer, epoch)

    test(model, device, test_loader)