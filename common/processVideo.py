import os
import cv2

## Save frames from video
def save_frames(root, file, output_dir):
    if not os.path.exists(output_dir + file):
        os.makedirs(output_dir + file)
    video = cv2.VideoCapture(root+file)
    if not video.isOpened():
        raise Exception("Could not open video")
    success,image = video.read()
    count = 0
    while success:
        output_file = output_dir + file + "/{count}.jpg"
        if not cv2.imwrite(output_file, image):
            raise Exception("Could not write image")
        success,image = video.read()
        count += 1
        
## Get frames from video
def get_frames(path):
    frames = [] 
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise Exception("Could not open video")
    success,image = video.read()
    while success:
        frames.append(image)
        success,image = video.read()
    return frames

## Get starting frame from video
def get_starting_frame(path):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise Exception("Could not open video")
    _,image = video.read()
    return image
