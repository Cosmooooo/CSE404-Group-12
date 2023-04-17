import os, csv, json

import numpy as np
import cv2
from tqdm import tqdm

from processVideo import get_starting_frame
from utils import *

## preprocess label data (center_x, center_y, scale, rotation, _, _) to desired label (min_x, min_y, max_x, max_y)
def data_preprocess(root, csv_file):
    data = []
    labels = dict()
    files = dict()
    for file_name in os.listdir(root):
        file = file_name.split(".")[0]
        _, video_id, clip_id, first_name, last_name = file.split("_")
        file = "_".join([first_name, last_name, video_id, clip_id])
        data.append(file)
        files[file] = os.path.join(root, file_name)

    with open (csv_file) as f:
        reader = csv.reader(f)
        row = next(reader)
        for row in reader:
            name, label = row[0].split(".")[0], row[1:]
            cx, cy, scale, rotation, _, _ = label
            left = float(cx) - float(scale) * 25
            top = float(cy) - float(scale) * 25
            right = float(cx) + float(scale) * 25
            bottom = float(cy) + float(scale) * 25
            label = [left, top, right, bottom]
            _, first_name, last_name, video_id, clip_id = name.split("_")
            labels["_".join([first_name, last_name, video_id, clip_id])] = label

    return data, labels, files

## convert raw data to json file
def raw_data_to_json(root, csv_file):
    data, labels, files = data_preprocess(root, csv_file)
    path_label = dict()

    for d in data:
        path_label[files[d]] = labels[d]

    with open("data/data.json", "w") as f:
        json.dump(path_label, f)

## Process JSON raw data to npz file with full information
def json_data_to_npz(json_path, save_path="data/data.npz"):
    f = open(json_path)
    data = json.load(f)

    images = []
    bbox_labels = []
    rois = []
    bbox_clses = []
    bbox_transforms = []

    for path, bbox_label in tqdm(data.items(), desc="Processing raw data"):
        image = get_starting_frame(path)
        image_size = image.shape[:2]

        # get bbox label and rois
        bbox_label = np.array(bbox_label, dtype=np.float32).reshape(-1, 4)
        roi = selective_search(image)
        # convert pixel value to scale value
        bbox_label = scale_bbox(image_size, bbox_label)
        roi = scale_bbox(image_size, roi)

        # get bbox_cls by ious
        ious = calculate_iou(roi, bbox_label)
        index = np.where(ious > 0.5)
        bbox_cls = np.zeros((len(roi), 1))
        bbox_cls[index] = 1

        # get bbox_transform
        bbox_transform = calculate_transform(roi, bbox_label)

        # resize image to shape (3, 224, 224)
        image = cv2.resize(image, (224, 224))
        image = image.transpose(2, 0, 1)

        images.append(image)
        bbox_labels.append(bbox_label)
        rois.append(roi)
        bbox_clses.append(bbox_cls)
        bbox_transforms.append(bbox_transform)

    images = np.array(images)
    bbox_labels = np.array(bbox_labels)
    rois = np.array(rois)
    bbox_clses = np.array(bbox_clses)
    bbox_transforms = np.array(bbox_transforms)
        
    print("Images shape", images.shape)
    print("Bbox labels shape", bbox_labels.shape)
    print("Rois shape", rois.shape)
    print("Bbox clses shape", bbox_clses.shape)
    print("Bbox transforms shape", bbox_transforms.shape)

    # save processed data
    np.savez(save_path, images=images, bbox_labels=bbox_labels, rois=rois, bbox_clses=bbox_clses, bbox_transforms=bbox_transforms)
    f.close()

json_data_to_npz("data/data.json")