import numpy as np
import cv2
from utils import undistort
from config import IMG_SHAPE

def draw(img, warped, invM, poly_param, curve_rad, offset, get_poly_points, mtx, dist):
    undist = undistort(img, mtx, dist)
    warp_zero = np.zeros_like(warped[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    left_fit = poly_param[0]
    right_fit = poly_param[1]
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(left_fit, right_fit)
    pts_left = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([plot_xright, plot_yright])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 220, 110))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 255, 255), thickness=10)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 255, 255), thickness= 10)
    unwarped = cv2.warpPerspective(color_warp, invM, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    out = cv2.addWeighted(undist, 1, unwarped, 0.4, 0)
    
    # The B term determines the base slope of the curve . Derivating the equation of the line gives us 2ay + b , this means that as y increases how much the line deviates from the center
    # 2ay term also contributes to the slope but b term is base slope term it tells us initial slope of the line .
    # so if b>0 it means that as we increase y the slope increases but as we know y increases from top to bottom but car is moving from bottom to top like going forward so in this case we conclude that when b>0 the curve is going left
    # same with b=0 straight and b<0 right curve as told above    
    # Now we are averaging both lane slopes to get general road slope idea
    
    if (left_fit[1] + right_fit[1]) / 2 > 0.05: # if average slope is greater than 0.05 then it is left turn 
        # we took a threshold of 0.05 rather than 0
        text = 'Left turn, curve radius: {:04.2f} m'.format(curve_rad)
    elif (left_fit[1] + right_fit[1]) / 2 < -0.05: # if average slope is less than -0.05 then it is right turn
        text = 'Right turn, curve radius: {:04.2f} m'.format(curve_rad)
    else:
        text = 'Straight'
        
   
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
    cv2.rectangle(out, (30, 35), (30 + text_size[0] + 20, 65), (0, 0, 0), -1)
    
    cv2.putText(out, text, (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    
    direction = ''
    if offset > 0:
        direction = 'left' # car is to the left of the center of the lane , should move right
    elif offset < 0:
        direction = 'right' # car is to the right of the center of the lane , should move left
    text = '{:0.1f} cm {} of center'.format(abs(offset) * 100, direction)
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
    cv2.rectangle(out, (30, 75), (30 + text_size[0] + 20, 105), (0, 0, 0), -1)
    
    cv2.putText(out, text, (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    
    return out