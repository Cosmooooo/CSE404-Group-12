import cv2
import os

## Save frames from video
def save_frames(root, file, output_dir):
    name = file.split(".")[0]
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
def get_frames(root, file):
    frames = [] 
    video = cv2.VideoCapture(root+file)
    if not video.isOpened():
        raise Exception("Could not open video")
    success,image = video.read()
    while success:
        frames.append(image)
        success,image = video.read()
    return frames

## Get starting frame from video
def get_starting_frame(root, file):
    video = cv2.VideoCapture(root+file)
    if not video.isOpened():
        raise Exception("Could not open video")
    _,image = video.read()
    return image
