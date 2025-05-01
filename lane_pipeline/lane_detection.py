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
        left_fit, right_fit = avg_params[0], avg_params[1]
        ret = False
    return ret, np.array([left_fit, right_fit]), np.array([left_fit, right_fit])

def visualize_polyfit(binary, left_fit, right_fit):
    """
    Visualize the polynomial fits with sliding windows on the binary image
    
    Args:
        binary: Binary image
        left_fit: Polynomial coefficients for the left lane
        right_fit: Polynomial coefficients for the right lane
        
    Returns:
        visualization: RGB image with sliding windows and polyfit visualized
    """
    
    out_img = np.dstack((binary*255, binary*255, binary*255)).astype(np.uint8)
    
    
    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    lft_lane = np.argmax(histogram[:midpoint])
    rgt_lane = np.argmax(histogram[midpoint:]) + midpoint
    
    y = binary.shape[0]
    box_height = 30
    margin = 40
    
    # Draw the sliding windows
    window_count = (binary.shape[0] // box_height) + (1 if binary.shape[0] % box_height > 0 else 0)
    
    
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(left_fit, right_fit)
    
    # For visualizing where the polynomial fit goes
    left_pts = []
    right_pts = []
    
    
    for window in range(window_count):
        win_y_top = y - (window + 1) * box_height
        win_y_top = max(0, win_y_top)
        win_y_bottom = y - window * box_height
        

        if win_y_bottom > 0:

            y_mid = (win_y_top + win_y_bottom) // 2
            left_x = left_fit[0] * y_mid**2 + left_fit[1] * y_mid + left_fit[2]
            right_x = right_fit[0] * y_mid**2 + right_fit[1] * y_mid + right_fit[2]
            
            
            left_x = max(0, min(binary.shape[1]-1, int(left_x)))
            right_x = max(0, min(binary.shape[1]-1, int(right_x)))
            
            left_pts.append((left_x, y_mid))
            right_pts.append((right_x, y_mid))
            
            left_x_low = max(0, left_x - margin)
            left_x_high = min(binary.shape[1]-1, left_x + margin)
            cv2.rectangle(out_img, 
                         (left_x_low, win_y_top), 
                         (left_x_high, win_y_bottom), 
                         (0, 255, 255), 3)  # Yellow for windows
            
            right_x_low = max(0, right_x - margin)
            right_x_high = min(binary.shape[1]-1, right_x + margin)
            cv2.rectangle(out_img, 
                         (right_x_low, win_y_top), 
                         (right_x_high, win_y_bottom), 
                         (0, 255, 255), 3)  # Yellow for windows
    
    for x, y in zip(plot_xleft, plot_yleft):
        cv2.circle(out_img, (x, y), 5, (0, 0, 255), -1)  
    
    for x, y in zip(plot_xright, plot_yright):
        cv2.circle(out_img, (x, y), 5, (0, 255, 0), -1)  
    
    # Draw the polynomial fit lines
    for i in range(len(plot_yleft) - 1):
        cv2.line(out_img, (plot_xleft[i], plot_yleft[i]), 
                 (plot_xleft[i+1], plot_yleft[i+1]), (255, 0, 0), 3)
    
    for i in range(len(plot_yright) - 1):
        cv2.line(out_img, (plot_xright[i], plot_yright[i]), 
                 (plot_xright[i+1], plot_yright[i+1]), (255, 0, 0), 3)
    
    # Show polynomial equation with proper superscript format
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    left_eq = f"Left: {left_fit[0]:.4f}y^2+{left_fit[1]:.4f}y+{left_fit[2]:.1f}"
    right_eq = f"Right: {right_fit[0]:.4f}y^2+{right_fit[1]:.4f}y+{right_fit[2]:.1f}"
    
    cv2.rectangle(out_img, (5, 5), (550, 95), (0, 0, 0), -1)
    
    cv2.putText(out_img, left_eq, (10, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_img, right_eq, (10, 80), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    return out_img
