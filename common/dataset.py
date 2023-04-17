import torch
from torch.utils.data import Dataset
import numpy as np 

class YTDataset(Dataset):
    def __init__(self, npz_path="data/data.npz"):
        data = np.load(npz_path)

        self.images = torch.Tensor(data["images"])
        self.bbox_labels = torch.Tensor(data["bbox_labels"])
        self.rois = torch.Tensor(data["rois"])
        self.bbox_clses = torch.Tensor(data["bbox_clses"]).long()
        self.bbox_transforms = torch.Tensor(data["bbox_transforms"])
                         
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.bbox_labels[idx], self.rois[idx], self.bbox_clses[idx], self.bbox_transforms[idx]
    
if __name__ == "__main__":
    dataset = YTDataset()
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    print(len(dataloader.dataset))
    for i, (images, bbox_labels, rois, bbox_clses, bbox_transforms) in enumerate(dataloader):
        print(images.shape)
        print(bbox_labels.shape)
        print(rois.shape)
        print(bbox_clses.shape)
        print(bbox_transforms.shape)
        break
