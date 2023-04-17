import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

## select 128 regions of interest using selective search
def selective_search(img, mode='f'):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if mode == 'f':
        ss.switchToSelectiveSearchFast()
    elif mode == 'q':
        ss.switchToSelectiveSearchQuality()
    rects = ss.process().astype(np.float32)

    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects[:128]

## convert bounding box to scale value (0 ~ 1)
def scale_bbox(image_size, bboxes):
    h, w = image_size
    bboxes[:, 0] /= w
    bboxes[:, 1] /= h
    bboxes[:, 2] /= w
    bboxes[:, 3] /= h
    return bboxes

## calculate the transform from rois to bbox_labels
def calculate_transform(rois, bbox_labels):
    rois_width = rois[:, 2] - rois[:, 0]
    rois_height = rois[:, 3] - rois[:, 1]
    rois_center_x = rois[:, 0] + rois_width / 2
    rois_center_y = rois[:, 1] + rois_height / 2

    bbox_labels_width = bbox_labels[:, 2] - bbox_labels[:, 0]
    bbox_labels_height = bbox_labels[:, 3] - bbox_labels[:, 1]
    bbox_labels_center_x = bbox_labels[:, 0] + bbox_labels_width / 2
    bbox_labels_center_y = bbox_labels[:, 1] + bbox_labels_height / 2

    tx = (bbox_labels_center_x - rois_center_x) / rois_width
    ty = (bbox_labels_center_y - rois_center_y) / rois_height
    tw = np.log(bbox_labels_width / rois_width)
    th = np.log(bbox_labels_height / rois_height)

    bbox_transform = np.stack([tx, ty, tw, th], axis=1)
    return bbox_transform

def transform_to(rois, bbox_transforms):
    rois_width = rois[:, 2] - rois[:, 0]
    rois_height = rois[:, 3] - rois[:, 1]
    rois_center_x = rois[:, 0] + rois_width / 2
    rois_center_y = rois[:, 1] + rois_height / 2

    tx = bbox_transforms[:, 0]
    ty = bbox_transforms[:, 1]
    tw = bbox_transforms[:, 2]
    th = bbox_transforms[:, 3]
 
    bbox_width = np.exp(tw) * rois_width
    bbox_height = np.exp(th) * rois_height
    bbox_center_x = tx * rois_width + rois_center_x
    bbox_center_y = ty * rois_height + rois_center_y

    bbox_x1 = bbox_center_x - bbox_width / 2
    bbox_y1 = bbox_center_y - bbox_height / 2
    bbox_x2 = bbox_center_x + bbox_width / 2
    bbox_y2 = bbox_center_y + bbox_height / 2

    bbox = np.stack([bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=1)
    return bbox

## calculate ious of rois and bbox_labels
def calculate_iou(rois, bbox_labels):
    rois_area = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
    bbox_labels_area = (bbox_labels[:, 2] - bbox_labels[:, 0]) * (bbox_labels[:, 3] - bbox_labels[:, 1])

    x1 = np.maximum(rois[:, 0], bbox_labels[:, 0])
    y1 = np.maximum(rois[:, 1], bbox_labels[:, 1])
    x2 = np.minimum(rois[:, 2], bbox_labels[:, 2])
    y2 = np.minimum(rois[:, 3], bbox_labels[:, 3])

    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)

    union_areas = w * h
    total_areas = rois_area + bbox_labels_area

    ious = union_areas / (total_areas - union_areas)
    return ious

## non-maximum suppression
# def nms(bboxes, scores, iou_threshold=0.3, score_threhold=0.3):
#     bboxes = bboxes[scores >= score_threhold]
#     scores = scores[scores >= score_threhold]

#     index = np.argsort(scores)[::-1]

#     bboxes = bboxes[index]
#     scores = scores[index]

#     nms_bboxes = []
#     nms_scores = []

#     while len(scores) > 0:
#         max_bbox = bboxes[0]
#         max_score = scores[0]

#         nms_bboxes.append(max_bbox)
#         nms_scores.append(max_score)

#         ious = calculate_iou(bboxes, max_bbox[np.newaxis, :])

#         thres_point = np.where(ious <= iou_threshold)[0]

#         bboxes = bboxes[thres_point]
#         scores = scores[thres_point]

#     nms_bboxes = np.stack(nms_bboxes, axis=0)
#     nms_scores = np.stack(nms_scores, axis=0)
#     return nms_bboxes, nms_scores

def plot(title, sub_titles, train_result, validate_result):
    if not os.path.exists('results'):
        os.makedirs('results')
    
    fig, axs = plt.subplots(len(train_result), figsize=(10, 10))
    fig.suptitle(title)

    for i, (train, validate) in enumerate(zip(train_result, validate_result)):
        axs[i].set_title(sub_titles[i])
        axs[i].set_ylim(0, max(max(train), max(validate)))
        axs[i].plot(train, label='train')
        axs[i].plot(validate, label='validate')
        axs[i].legend()

    plt.savefig('results/{}.png'.format(title))
