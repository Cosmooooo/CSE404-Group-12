import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

from common.YTCelebrityDatasetFirstFrame import YTCelebrityDatasetFirstFrame
from common.process import draw_square_by_points, get_bounding_box
from ResNet import ResNet
from SimpleCNN import SimpleCNN
from Yolo import Yolo
from VGG import VGG
from LeNet import LeNet
from shared import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser(**params):
    def str_to_bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ("yes", "y", "true", "t", "1", 1):
            return True
        elif s.lower() in ("no", "n", "false", "f", "0", 0):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        
    parser = argparse.ArgumentParser(**params)
    
    # hyperparameters
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--validate', type=str_to_bool, default=False, help='validate or not')

    # data path
    parser.add_argument('--root', type=str, default="/media/cosmo/Dataset/YTCelebrity/ytcelebrity/", help='root path of dataset')
    parser.add_argument('--csv', type=str, default="celebrity.csv", help='csv path of dataset')

    # load model
    parser.add_argument('--model', type=str, default="lenet", help='save checkpoint path')
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint path')
    
    return parser

def main():
    parser = get_parser()
    opt, _ = parser.parse_known_args()

    root = opt.root
    csv_path = opt.csv
    
    batch_size = opt.batch_size
    epoch = opt.epoch
    learning_rate = opt.lr
    validate = opt.validate

    def get_model(s):
        models = {"simplecnn": SimpleCNN(), "vgg": VGG(), "resnet": ResNet(), "yolo": Yolo(), "lenet": LeNet()}
        if s.lower() in models:
            return models[s.lower()]
        else:
            raise argparse.ArgumentTypeError("Model not found.")
        
    model = get_model(opt.model).to(device)
    ckpt = opt.ckpt

    if ckpt is not None:
        model = load_model(ckpt)

    dataset = YTCelebrityDatasetFirstFrame(root, csv_path)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sys.stdout.write("\nSplitting...")
    generator = torch.Generator().manual_seed(41)
    training, validation, testing = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=True, num_workers=4)
    
    sys.stdout.write("\nTraining...")
    
    iou_track, l1_track, val_iou_track, val_l1_track = train(model, device, train_loader, optimizer, epoch, validate=validate, validation_loader=validation_loader)
    
    fig, axs = plt.subplots(2)
    fig.suptitle(f'{model.__class__.__name__}')
    axs[0].set_title('IOU Accuracy')
    axs[0].plot(iou_track, label="Train_iou")
    axs[0].set_ylim(0, 1)
    axs[1].set_title('L1 Loss')
    axs[1].plot(l1_track, label="Train_l1")
    if validate:
        axs[0].plot(val_iou_track, label="Val_iou")
        axs[1].plot(val_l1_track, label="Val_l1")
    axs[0].legend()
    axs[1].legend()
    plt.savefig(f'results/{model.__class__.__name__}.png')
    save_model(model, f'checkpoints/{model.__class__.__name__}_{epoch}.pth')

    sys.stdout.write("\nTesting...")
    test_iou, test_l1 = test(model, device, test_loader)

# main()
# from data.data import get_starting_frame
# from data.process import draw_square_by_label
# import cv2


# root = "/media/cosmo/Dataset/YTCelebrity/ytcelebrity/"
# video = "1905_03_006_vladimir_putin.avi"
# image = get_starting_frame(root, video)
# cv2.imwrite("test.jpg", image)
# for model in (LeNet(), VGG(), ResNet(), Yolo()):
#     model = load_model(model, f"checkpoints/{model.__class__.__name__}_150.pth")
#     prediction = predict_image(model, image)
#     image = get_starting_frame(root, video)
#     image = draw_square_by_label(image, *prediction.numpy())
#     cv2.imwrite(f"results/{model.__class__.__name__}_test.png", image)
