import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from utils import plot_images
from calibration import load_or_calibrate
from perspective import preprocess_image
from thresholding import get_binary_image
from lane_detection import polyfit_sliding_window, get_poly_points
from metrics import compute_mppx, compute_offset_from_center, compute_curvature
from draw import draw
from config import VIDEO_INPUT, VIDEO_OUTPUT
import os
from config import CAMERA_CALIB_PATH
import pickle
import glob
import matplotlib.image as mpimg
# Pipeline state
cache = np.array([])
poly_param = None

# Calibration (if camera matrix and distortion coeff not loaded then it finds it along with meter along x and y  )
mtx, dist , x_mppx, y_mppx = load_or_calibrate()

def pipeline(img):
    global cache, poly_param
    result = np.copy(img)
    warped, (M, invM) = preprocess_image(img, mtx, dist)
    try:
        binary = get_binary_image(warped)
        ret, poly_param, _ = polyfit_sliding_window(binary, cache)
        if ret:
            cache = np.array([poly_param])
        else:
            if len(poly_param) == 0:
                return img
        left_curverad, right_curverad = compute_curvature(poly_param, y_mppx, x_mppx, get_poly_points)
        offset = compute_offset_from_center(poly_param, x_mppx, get_poly_points)
        result = draw(img, warped, invM, poly_param, (left_curverad + right_curverad) / 2, offset, get_poly_points, mtx, dist)
        return result
    except Exception as e:
        print(e)
        return img

def main():
    video_input = VIDEO_INPUT
    video_output = VIDEO_OUTPUT
    video_clip = VideoFileClip(video_input)
    processed_clip = video_clip.fl_image(pipeline)
    processed_clip.write_videofile(video_output, audio=False)
    
    '''# for image's 
    for img in test_img_paths:
     result = pipeline(mpimg.imread(img))
     plot_images([(result, "test")])
     plt.show()'''
        

if __name__ == '__main__':
    main() 