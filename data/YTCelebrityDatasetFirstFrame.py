import torch
from torch.utils.data import Dataset
from data.data import get_starting_frame
from torchvision import transforms
import numpy as np
import csv
import os

## Dataset for YTCelebrity dataset
class YTCelebrityDatasetFirstFrame(Dataset):
    def __init__(self, root_path, csv_path, transform=None):
        self.root = root_path
        self.data = []
        self.image = dict()
        self.label = dict()
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((360, 360)),
                transforms.ToTensor()
            ])
        for file_name in os.listdir(self.root):
            file = file_name.split(".")[0]
            _, video_id, clip_id, first_name, last_name = file.split("_")
            file = "_".join([first_name, last_name, video_id, clip_id])
            self.data.append(file)
            image = get_starting_frame(self.root, file_name)
            self.image[file] = image

        with open (csv_path) as f:
            reader = csv.reader(f)
            row = next(reader)
            for row in reader:
                name, label = row[0].split(".")[0], row[1:5]
                _, first_name, last_name, video_id, clip_id = name.split("_")
                self.label["_".join([first_name, last_name, video_id, clip_id])] = torch.from_numpy(np.asarray(label).astype(np.float32))
        for file in self.data:
            self.image[file], self.label[file] = self.transform(self.image[file]), self.label[file]
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.image[self.data[idx]]
        label = self.label[self.data[idx]]
        return image, label
