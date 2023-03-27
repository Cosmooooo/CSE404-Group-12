import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.YTCelebrityDataset import YTCelebrityDataset
from data.YTCelebrityDatasetFirstFrame import YTCelebrityDatasetFirstFrame
from models.shared import *
from models.Zongyuan import Zongyuan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
epoch = 100
root = "/media/cosmo/Dataset/YTCelebrity/ytcelebrity/"
csv_path = "celebrity.csv"
batch_size = 64

def main():
    dataset = YTCelebrityDatasetFirstFrame(root, csv_path)

    print("\nsplitting...\n")
    generator = torch.Generator().manual_seed(42)
    training, testing = torch.utils.data.random_split(dataset, [0.7, 0.3], generator=generator)

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=True, num_workers=1)

    model = Zongyuan().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    print("\ntraining...\n")
    for i in range(epoch):
        loss_track, total_loss = train(model, device, train_loader, optimizer)
        print(f"Epoch: {i} total loss: {total_loss}, mean loss: {total_loss / len(loss_track) / batch_size}")
    save_model(model, f'{model.__class__.__name__}_{epoch}.pth')
    print("\ntesting...\n")
    loss_track, total_loss = test(model, device, test_loader)
    print(f"Test total loss: {total_loss}, mean loss: {total_loss / len(loss_track) / batch_size}")

main()
