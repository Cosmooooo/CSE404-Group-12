import sys, json
sys.path.append("/home/cosmo/Desktop/cse404/")

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common.utils import *
from common.data import get_starting_frame

class boundingBoxDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.image = []
        self.label = []

        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((227, 227)),
                transforms.ToTensor()
            ])

        f = open(json_path)
        data = json.load(f)

        for path, label in tqdm(data.items(), desc="Loading bounding box regression data"):
            image = get_starting_frame(path)

            count = 0
            for rect in selective_search(image):
                iou, area = check_iou_area(rect, label)
                if iou >= 0.7 and area >= 0.4:
                    region = get_region(image, rect)
                    self.image.append(self.transform(region))  

                    x, y, s, r = rect
                    lx, ly, ls, lr = label
                    dx = (lx - x) / 5
                    dy = (ly - y) / 5
                    ds = ls - s
                    dr = (lr - r) * 3
                    self.label.append(torch.Tensor([dx, dy, ds, dr]))
                    count += 1
                if count >= 10:
                    break
        f.close()
                           
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.image[idx], self.label[idx]
