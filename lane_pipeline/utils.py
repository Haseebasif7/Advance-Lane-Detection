import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import glob

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    l.sort(key=alphanum_key)

def plot_images(data, layout='row', cols=2, figsize=(20, 12)):
    rows = math.ceil(len(data) / cols)
    f, ax = plt.subplots(figsize=figsize)
    if layout == 'row':
        for idx, d in enumerate(data):
            img, title = d
            plt.subplot(rows, cols, idx+1)
            plt.title(title, fontsize=20)
            plt.axis('off')
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            elif len(img.shape) == 3:
                plt.imshow(img)
    elif layout == 'col':
        counter = 0
        for r in range(rows):
            for c in range(cols):
                img, title = data[r + rows*c]
                plt.subplot(rows, cols, counter+1)
                plt.title(title, fontsize=20)
                plt.axis('off')
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')
                elif len(img.shape) == 3:
                    plt.imshow(img)
                counter += 1
    return ax

def capture_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    print('Starting frame capture...')
    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if success:
            cv2.imwrite(frames_dir + 'frame{:02}.jpg'.format(count), frame)
            count += 1
    print('Completed!') 

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)