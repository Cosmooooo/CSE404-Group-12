import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.YTCelebrityDataset import YTCelebrityDataset
from models.model import Model, train, test
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
epoch = 10
root = "/media/cosmo/Dataset/YTCelebrity/ytcelebrity/"
csv_path = "celebrity.csv"

def main():
    dataset = YTCelebrityDataset(root, csv_path)

    new_dataset = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    print("\nconverting...\n")
    for _, (frames, label) in enumerate(dataset):
        starting_frame = transform(np.asarray(frames[0]))
        starting_label = [float(l) for l in label]
        new_dataset.append((starting_frame, torch.from_numpy(np.asarray(starting_label[:4]))))
    
    print("\nsplitting...\n")
    generator = torch.Generator().manual_seed(42)
    training, testing = torch.utils.data.random_split(dataset, [0.7, 0.3], generator=generator)

    train_loader = DataLoader(training, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(testing, batch_size=1, shuffle=True, num_workers=1)

    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    print("\ntraining...\n")
    train(model, device, train_loader, optimizer)
    print("\ntesting...\n")
    test(model, device, test_loader)

main()
