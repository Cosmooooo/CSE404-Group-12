import json

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common.data import get_starting_frame
from common.utils import selective_search, check_iou_area

## Dataset for YTCelebrity dataset
class FastRCNNDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.image = []
        self.rois = []
        self.cls_label = []
        self.reg_label = []

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        f = open(json_path)
        data = json.load(f)

        for path, label in tqdm(data.items(), desc="Loading FastRCNN data"):
            image = get_starting_frame(path)
            self.image.append(self.transform(image)) 
            rois = selective_search(image)
            
            for roi in rois[:20]:
                iou, area = check_iou_area(roi, label)
                if iou > 0.5 and area > 0.2:
                    self.cls_label.append(1)   
                else:
                    self.cls_label.append(0)

                x, y, s, r = roi
                lx, ly, ls, lr = label
                dx = (lx - x) / 5
                dy = (ly - y) / 5
                ds = ls - s
                dr = (lr - r) * 3
                self.reg_label.append(torch.Tensor([dx, dy, ds, dr]))
                self.rois.append(roi)
                break
            break
        f.close()
         
    def __len__(self):
        return len(self.cls_label)

    def __getitem__(self, idx):
        return self.image[idx//20], self.rois[idx], self.cls_label[idx//20], self.reg_label[idx//20]
    