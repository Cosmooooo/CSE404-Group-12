import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2

from common.dataset import YTDataset
import FastRCNN.utils as fast_rcnn
import RCNN.utils as rcnn
import Resnet.utils as resnet
import VGG.utils as vgg
import LeNet.utils as lenet
import Yolobackbone.utils as yolo
import SimpleCNN.utils as simplecnn

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

    # file
    parser.add_argument('--file', type=str, help="video file to predict")

    # load model
    parser.add_argument('--model', type=str, default="lenet", help='model includes simplecnn, lenet, vgg, resnet, yolo, rcnn, fastrcnn')
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint path')

    parser.add_argument('--mode', type=str, default="train", help='task mode: train, test, predict')
    parser.add_argument('--save', type=str, default=None, help='save path for checkpoint file')
    
    return parser

def main():
    parser = get_parser()
    opt, _ = parser.parse_known_args()

    batch_size = opt.batch_size
    epoch = opt.epoch
    learning_rate = opt.lr

    def get_model_mode(s):
        models = {"simplecnn": simplecnn, "lenet": lenet, "vgg": vgg, "resnet": resnet, "yolo": yolo, "rcnn": rcnn, "fastrcnn": fast_rcnn}
        if s.lower() in models:
            return models[s.lower()]
        else:
            raise argparse.ArgumentError("Model not found.")
        
    def get_task_mode(s):
        modes = {"train", "test", "predict"}
        if s.lower() in modes:
            return s.lower()
        else:
            raise argparse.ArgumentError("Please select available mode.")
        
    model_mode = get_model_mode(opt.model)
    mode = get_task_mode(opt.mode)

    dataset = YTDataset()
    generator = torch.Generator().manual_seed(41)

    training, validation, testing = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam

    model = None

    if mode == "train":
        model = model_mode.train(train_loader, validation_loader, optimizer, epoch, learning_rate)
        model_mode.test(model, test_loader)
        torch.save(model.state_dict(), f"checkpoints/{model.__class__.__name__}.pth" if not opt.save else opt.save)

    elif mode == "test":
        if opt.ckpt:
            model = model_mode.load(opt.ckpt)
            model_mode.test(model, test_loader)
        raise Exception("No checkpoint file provided")
    
    elif mode == "predict":
        if not opt.ckpt:
            raise Exception("No checkpoint file provided")
        if not opt.file:
            raise Exception("No video file provided")
        
        model = model_mode.load(opt.ckpt)
        video = cv2.VideoCapture(opt.file)

        if not video.isOpened():
            raise Exception("Could not open video")
        
        success,image = video.read()

        frame = int(1000/video.get(cv2.CAP_PROP_FPS))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter("predicted.mp4", cv2.VideoWriter_fourcc(*'mp4v'), frame, (frame_width, frame_height))

        while success:
            image, bboxes = model_mode.predict_image(model, image)
            for bbox in bboxes:
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            out.write(image)
            cv2.imshow("video", image)
            success,image = video.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()
        
main()