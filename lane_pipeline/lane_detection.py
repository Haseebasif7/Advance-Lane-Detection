import numpy as np
import cv2
from config import IMG_SHAPE
from thresholding import get_binary_image

def get_poly_points(left_fit, right_fit):
    ysize, xsize = IMG_SHAPE
    plot_y = np.linspace(0, ysize-1, ysize)
    plot_xleft = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    plot_xright = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
    plot_xleft = plot_xleft[(plot_xleft >= 0) & (plot_xleft <= xsize - 1)]
    plot_xright = plot_xright[(plot_xright >= 0) & (plot_xright <= xsize - 1)]
    plot_yleft = np.linspace(ysize - len(plot_xleft), ysize - 1, len(plot_xleft))
    plot_yright = np.linspace(ysize - len(plot_xright), ysize - 1, len(plot_xright))
    return (plot_xleft.astype(int), plot_yleft.astype(int), plot_xright.astype(int), plot_yright.astype(int))

def check_validity(left_fit, right_fit):
    if left_fit is None or right_fit is None:
        return False
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(left_fit, right_fit)
    y1 = IMG_SHAPE[0] - 1
    y2 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.35)
    y3 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.75)
    x1l = left_fit[0]  * (y1**2) + left_fit[1]  * y1 + left_fit[2]
    x2l = left_fit[0]  * (y2**2) + left_fit[1]  * y2 + left_fit[2]
    x3l = left_fit[0]  * (y3**2) + left_fit[1]  * y3 + left_fit[2]
    x1r = right_fit[0] * (y1**2) + right_fit[1] * y1 + right_fit[2]
    x2r = right_fit[0] * (y2**2) + right_fit[1] * y2 + right_fit[2]
    x3r = right_fit[0] * (y3**2) + right_fit[1] * y3 + right_fit[2]
    x1_diff = abs(x1l - x1r)
    x2_diff = abs(x2l - x2r)
    x3_diff = abs(x3l - x3r)
    min_dist_y1 = 480
    max_dist_y1 = 730
    min_dist_y2 = 280
    max_dist_y2 = 730
    min_dist_y3 = 140
    max_dist_y3 = 730
    if (x1_diff < min_dist_y1) | (x1_diff > max_dist_y1) | \
        (x2_diff < min_dist_y2) | (x2_diff > max_dist_y2) | \
        (x3_diff < min_dist_y3) | (x3_diff > max_dist_y3):
        return False
    y1left_dx  = 2 * left_fit[0]  * y1 + left_fit[1]
    y3left_dx  = 2 * left_fit[0]  * y3 + left_fit[1]
    y1right_dx = 2 * right_fit[0] * y1 + right_fit[1]
    y3right_dx = 2 * right_fit[0] * y3 + right_fit[1]
    norm1 = abs(y1left_dx - y1right_dx)
    norm2 = abs(y3left_dx - y3right_dx)
    thresh = 0.6
    if (norm1 >= thresh) | (norm2 >= thresh):
        return False
    return True

def polyfit_sliding_window(binary, cache):
    '''
    Detect lane lines in a thresholded binary image using the sliding window technique (adapted from user version),
    but return (ret, np.array([left_fit, right_fit]), np.array([left_fit, right_fit])) for pipeline compatibility.
    '''
    ret = True
    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    lft_lane = np.argmax(histogram[:midpoint])
    rgt_lane = np.argmax(histogram[midpoint:]) + midpoint

    y = binary.shape[0]
    box_height = 30
    margin = 40
    lx = []
    rx = []
    ly = []
    ry = []

    while y > 0:
        # Left window
        left_x_low = max(lft_lane - margin, 0)
        left_x_high = min(lft_lane + margin, binary.shape[1])
        img = binary[max(y - box_height, 0):y, left_x_low:left_x_high]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                lx.append(left_x_low + cx)
                ly.append(y)
                lft_lane = left_x_low + cx

        # Right window
        right_x_low = max(rgt_lane - margin, 0)
        right_x_high = min(rgt_lane + margin, binary.shape[1])
        img = binary[max(y - box_height, 0):y, right_x_low:right_x_high]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                rx.append(right_x_low + cx)
                ry.append(y)
                rgt_lane = right_x_low + cx

        y -= box_height  # move window up

    # Fit 2nd order polynomials if enough points
    min_lane_pts = 10
    left_fit, right_fit = None, None
    if len(lx) >= min_lane_pts and len(rx) >= min_lane_pts:
        left_fit = np.polyfit(ly, lx, 2)
        right_fit = np.polyfit(ry, rx, 2)
    valid = check_validity(left_fit, right_fit)
    if not valid:
        if len(cache) == 0:
            return False, np.array([]), np.array([])
        avg_params = np.mean(cache, axis=0)
        # Cache has left fit and right fit of previous lane stored 
        # so we take mean of all left fit coefficients and right fit 
        # cache has (left_fit , right_fit) where left_fit = (A,B,C) for left lane and right_fit has (A,B,C) for right lane
        # and there are multiple pair of (left_fit , right_fit) depending on previous road frame
        left_fit, right_fit = avg_params[0], avg_params[1]
        ret = False
    return ret, np.array([left_fit, right_fit]), np.array([left_fit, right_fit])
