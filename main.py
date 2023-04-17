import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from common.dataset import YTDataset
import FastRCNN.utils as fast_rcnn
import RCNN.utils as rcnn
import Resnet.utils as resnet
import VGG.utils as vgg
import LeNet.utils as lenet
import Yolobackbone.utils as yolo

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
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # load model
    parser.add_argument('--model', type=str, default="lenet", help='model includes lenet, vgg, resnet, yolo, rcnn, fastrcnn')
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint path')

    parser.add_argument('--train', type=str_to_bool, default=False, help='train model')
    parser.add_argument('--save', type=str, default=None, help='save path for checkpoint file')
    
    return parser


def main():
    parser = get_parser()
    opt, _ = parser.parse_known_args()

    batch_size = opt.batch_size
    epoch = opt.epoch
    learning_rate = opt.lr

    def get_model_mode(s):
        models = {"lenet": lenet, "vgg": vgg, "resnet": resnet, "yolo": yolo, "rcnn": rcnn, "fastrcnn": fast_rcnn}
        if s.lower() in models:
            return models[s.lower()]
        else:
            raise argparse.ArgumentTypeError("Model not found.")
        
    model_mode = get_model_mode(opt.model)

    dataset = YTDataset()
    generator = torch.Generator().manual_seed(41)

    training, validation, testing = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam

    model = None

    if opt.train:
        model = model_mode.train(train_loader, validation_loader, optimizer, epoch, learning_rate)
        model_mode.test(model, test_loader)

        torch.save(model.state_dict(), f"checkpoints/{model.__class__.__name__}.pth" if not opt.save_path else opt.save_path)

    else:
        if opt.ckpt:
            model = model_mode.load(opt.ckpt)
            model_mode.test(model, test_loader)
        
main()