import cv2, sys
sys.path.append("/home/cosmo/Desktop/cse404/")

import torch
import numpy as np

from common.process import get_iou, get_area_fraction, get_bounding_box, estimate_iou

def selective_search(img, mode='f'):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if mode == 'f':
        ss.switchToSelectiveSearchFast()
    elif mode == 'q':
        ss.switchToSelectiveSearchQuality()
    rects = ss.process().astype('float')

    rects[:, 0] += rects[:, 2] / 2
    rects[:, 1] += rects[:, 3] / 2
    rects[:, 2] = (rects[:, 2] + rects[:, 3]) / (2 * 50)
    rects[:, 3] = 0

    return rects[:2000]

def check_iou_area(rect, label):
    square_A = get_bounding_box(*rect)
    square_B = get_bounding_box(*label)
    return estimate_iou(square_A, square_B), get_area_fraction(square_A, square_B)

def get_region(image, rect):
    x, y, s, r = rect
    y_min = max(int(y - s * 25), 0)
    y_max = min(int(y + s * 25), image.shape[0] if isinstance(image, np.ndarray) else image.shape[1])
    x_min = max(int(x - s * 25), 0)
    x_max = min(int(x + s * 25), image.shape[1] if isinstance(image, np.ndarray) else image.shape[2])
    if isinstance(image, np.ndarray):
        return image[y_min : y_max, x_min : x_max, : ]
    if len(image.shape) == 3:
        return image[:, y_min : y_max, x_min : x_max]
    return image[:, :, y_min : y_max, x_min : x_max]

## non-maximum suppression
def nms(boxes, scores, threshold=0.3):
    boxes = np.array(boxes)
    scores = np.array(scores)

    index = np.argsort(scores)[::-1]
    boxes = boxes[index]
    scores = scores[index]
    
    keep = []
    while len(scores) > 0:
        keep.append(index[0])
        boxes = boxes[1:]
        scores = scores[1:]
        if len(scores) <= 0:
            break
        
        bounding_boxes = [get_bounding_box(*b) for b in boxes]
        index = []
        for i in range(1, len(bounding_boxes)):
            iou_score = estimate_iou(bounding_boxes[0], bounding_boxes[i])
            if iou_score < threshold:
                index.append(i)
        index = np.array(index, dtype=np.int32)

        boxes = boxes[index]
        scores = scores[index]
    return keep

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

