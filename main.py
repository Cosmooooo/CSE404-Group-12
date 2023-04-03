import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.YTCelebrityDatasetFirstFrame import YTCelebrityDatasetFirstFrame
from data.process import draw_square_by_points, get_bounding_box
from models.shared import *
from models.SimpleCNN import SimpleCNN
import matplotlib.pyplot as plt
import argparse, sys

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
        
    def get_model(s):
        if s.lower() in ("simplecnn"):
            return SimpleCNN()
        else:
            raise argparse.ArgumentTypeError("Model not found.")
        
    parser = argparse.ArgumentParser(**params)
    
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--validate', type=str_to_bool, default=False, help='validate or not')

    # data path
    parser.add_argument('--root', type=str, default="/media/cosmo/Dataset/YTCelebrity/ytcelebrity/", help='root path of dataset')
    parser.add_argument('--csv', type=str, default="celebrity.csv", help='csv path of dataset')

    # load model
    parser.add_argument('--model', type=get_model, default="simplecnn", help='save checkpoint path')
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

    ckpt = opt.ckpt
    model = opt.model.to(device)
    if ckpt is not None:
        model = load_model(ckpt)

    dataset = YTCelebrityDatasetFirstFrame(root, csv_path)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
    plt.savefig(f'{model.__class__.__name__}.png')
    save_model(model, f'{model.__class__.__name__}_{epoch}.pth')

    sys.stdout.write("\nTesting...")
    test_iou, test_l1 = test(model, device, test_loader)

main()
