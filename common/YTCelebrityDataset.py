from torch.utils.data import Dataset
import os
from common.data import get_frames
import csv

## Dataset for YTCelebrity dataset
class YTCelebrityDataset(Dataset):
    def __init__(self, dataset_path, csv_path):
        self.root = dataset_path
        self.data = []
        self.label = dict()
        self.file = dict()
        for file_name in os.listdir(dataset_path):
            file = file_name.split(".")[0]
            _, video_id, clip_id, first_name, last_name = file.split("_")
            file = "_".join([first_name, last_name, video_id, clip_id])
            self.data.append(file)
            self.file[file] = file_name

        with open (csv_path) as f:
            reader = csv.reader(f)
            row = next(reader)
            for row in reader:
                name, label = row[0].split(".")[0], row[1:]
                _, first_name, last_name, video_id, clip_id = name.split("_")
                self.label["_".join([first_name, last_name, video_id, clip_id])] = label
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = get_frames(self.root, self.file[self.data[idx]])
        label = self.label[self.data[idx]]
        return frames, label
