import numpy as np
import cv2
from config import IMG_SHAPE
from utils import undistort

def get_roi(img, vertices):
    vertices = np.array(vertices, ndmin=3, dtype=np.int32)
    if len(img.shape) == 3:
        fill_color = (255,) * 3
    else:
        fill_color = 255
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, vertices, fill_color)
    # black out irrelevant extra pixels 
    return cv2.bitwise_and(img, mask)

def warp_image(img, warp_shape, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, warp_shape, flags=cv2.INTER_LINEAR)
    return warped, M, invM

def preprocess_image(img, mtx, dist):
    ysize = img.shape[0]
    xsize = img.shape[1]
    undist = undistort(img, mtx, dist)
    # For perspective transform the src points and dst points vary according to camera and road orientation
    src = np.float32([
        (696,455),    
        (587,455), 
        (235,700),  
        (1075,700)
    ])
    dst = np.float32([
        (xsize - 350, 0),
        (350, 0),
        (350, ysize),
        (xsize - 350, ysize)
    ])
    warped, M, invM = warp_image(undist, (xsize, ysize), src, dst)
    # vertices for region of interest 
    vertices = np.array([
        [200, ysize],
        [200, 0],
        [1100, 0],
        [1100, ysize]
    ])
    roi = get_roi(warped, vertices)
    return roi, (M, invM) 

def p_undistort(img, mtx, dist):
    undistort(img,mtx,dist)