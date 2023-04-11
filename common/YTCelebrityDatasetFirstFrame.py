import json

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common.data import get_starting_frame

## Dataset for YTCelebrity dataset
class YTCelebrityDatasetFirstFrame(Dataset):
    def __init__(self, json_path, transform=None):
        self.image = []
        self.label = []
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((360, 360)),
                transforms.ToTensor()
            ])
        
        f = open(json_path)
        data = json.load(f)

        for path, label in tqdm(data.items(), desc="Loading YTcelevrity First Frame data"):
            image = get_starting_frame(path)
            self.image.append(self.transform(image))            
            self.label.append(torch.Tensor(label))
        f.close()
         
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.image[idx], self.label[idx]
    
if __name__ == "__main__":
    dataset = YTCelebrityDatasetFirstFrame("data/YTCelebrity/first_frame.json")
    
