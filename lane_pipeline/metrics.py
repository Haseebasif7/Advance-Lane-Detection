import numpy as np
from thresholding import get_binary_image
from config import IMG_SHAPE

def compute_mppx(img, dashed_line_loc):
    # Standard values of lane in America , UK etc 
    lane_width = 3.7
    dashed_line_len = 3.048
    if dashed_line_loc == 'left':
        y_top = 295
        y_bottom = 405
    elif dashed_line_loc == 'right':
        y_top = 395
        y_bottom = 495
    binary = get_binary_image(img)
    histogram = np.sum(binary[int(binary.shape[0] / 2):, :], axis=0)
    # same histogram method for left and right lane bottom pixel along width
    midpoint = np.int_(histogram.shape[0] / 2)
    x_left = np.argmax(histogram[:midpoint])
    x_right = np.argmax(histogram[midpoint:]) + midpoint
    x_mppx = lane_width / (x_right - x_left) # meter per pixel along width 
    y_mppx = dashed_line_len / (y_bottom - y_top) # meter per pixel along height
    return y_mppx, x_mppx

def compute_offset_from_center(poly_param, x_mppx, get_poly_points):
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(poly_param[0], poly_param[1])
    #Accessing the last value (bottom as car is assumed to be in bottom and we are finding right lane pixel value for that bottom y pixel ) of X pixel for corresponding y value 
    lane_center = (plot_xright[-1] + plot_xleft[-1]) / 2
    car_center = IMG_SHAPE[1] / 2
    offset = (lane_center - car_center) * x_mppx # offset the car is from centre of road in meters 
    return offset 

def compute_curvature(poly_param, y_mppx, x_mppx, get_poly_points):
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(poly_param[0], poly_param[1])
    #Y-position where you want to evaluate the curvature of the road, because that's where the car is.
    y_eval = np.max(plot_yleft) # Near the car bottom of it 
    left_fit_cr = np.polyfit(plot_yleft * y_mppx, plot_xleft * x_mppx, 2)
    right_fit_cr = np.polyfit(plot_yright * y_mppx, plot_xright * x_mppx, 2)
    # Now that I had fitted curve in terms of meters so that when i give any car position (y) in meter i can get how far left and right lane are in meter
    left_curverad = ((1 + (2*left_fit_cr[0]* y_eval*y_mppx + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*y_mppx + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # these are calculating radius of left and right lanes 
    return left_curverad, right_curverad 