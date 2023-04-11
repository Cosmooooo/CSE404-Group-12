import sys, json
sys.path.append("/home/cosmo/Desktop/cse404/")

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

from common.utils import *
from common.data import get_starting_frame

class classificationDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.image = []
        self.label = []
        self.positive = 0
        self.negative = 0

        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((227, 227)),
                transforms.ToTensor()
            ])

        f = open(json_path)
        data = json.load(f)

        for path, label in tqdm(data.items(), desc="Loading classification data"):
            image = get_starting_frame(path)
            rects = selective_search(image)
            
            rects = sorted(rects, key=lambda x: (check_iou_area(x, label)[0], check_iou_area(x, label)[1]), reverse=True)

            for i in range(6):
                region = get_region(image, rects[i])
                self.image.append(self.transform(region))
                self.label.append(1)
                self.positive += 1
            
            for i in range(6, 32, 5):
                region = get_region(image, rects[i])
                self.image.append(self.transform(region))
                self.label.append(1)
                self.positive += 1

            for i in range(100, len(rects), (len(rects) - 50) // 12):
                region = get_region(image, rects[i])
                self.image.append(self.transform(region))
                self.label.append(0)
                self.negative += 1

        self.label = torch.LongTensor(self.label)
        f.close()
                       
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.image[idx], self.label[idx]
    
# if __name__ == '__main__':
#     data = classificationDataset("/home/cosmo/Desktop/cse404/data.json")
#     for i, (img, label) in enumerate(zip(data.image, data.label)):
#         c_img = img.copy()
#         c_img = cv2.putText(c_img, str(label.item()), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#         cv2.imwrite(f"/home/cosmo/Desktop/test/{i}_{str(label.item())}.jpg", c_img)