import cv2
import numpy as np

def binary_threshold(img, low, high):    
    if len(img.shape) == 2:
        output = np.zeros_like(img)
        mask = (img >= low) & (img <= high)
    elif len(img.shape) == 3:
        output = np.zeros_like(img[:,:,0])
        mask = (img[:,:,0] >= low[0]) & (img[:,:,0] <= high[0]) \
            & (img[:,:,1] >= low[1]) & (img[:,:,1] <= high[1]) \
            & (img[:,:,2] >= low[2]) & (img[:,:,2] <= high[2])
    output[mask] = 1 # put 1 whereever its true else white 
    return output

def get_binary_image(img):
    # Seperating lane pixels from road using red intensity 
    R = img[:,:,0]
    R_max, R_mean = np.max(R), np.mean(R)
    R_low_white = min(max(150, int(R_max * 0.55), int(R_mean * 1.95)),230) # threshold
    R_binary = binary_threshold(R, R_low_white, 255)
    return  R_binary