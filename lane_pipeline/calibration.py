import cv2
import numpy as np
import glob
import pickle
import os
from config import CAMERA_CALIB_PATH, CAMERA_CAL_IMAGES
from perspective import preprocess_image
from metrics import compute_mppx
from utils import sort_nicely
import matplotlib.image as mpimg

def calibrate_camera():
    imgpaths = glob.glob(CAMERA_CAL_IMAGES)
    sort_nicely(imgpaths)
    image = cv2.imread(imgpaths[0])
    imshape = image.shape[:2]
    objpoints = []
    imgpoints = []
    nx = 9
    ny = 6
    objp = np.zeros([ny*nx, 3], dtype=np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for imgpath in imgpaths:
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imshape[::-1], None, None)
    return mtx, dist

def load_or_calibrate():
    if os.path.exists(CAMERA_CALIB_PATH):
        with open(CAMERA_CALIB_PATH, mode='rb') as f:
            data = pickle.load(f)
            mtx, dist = data['mtx'], data['dist']
            x_mppx, y_mppx = data['x_mppx'], data['y_mppx']
    else:
        mtx, dist = calibrate_camera()

        test_img_paths = glob.glob('test_images/test*.jpg')
        sort_nicely(test_img_paths)
        img1 = mpimg.imread(test_img_paths[0])
        img2 = mpimg.imread(test_img_paths[1])

        warped1, _ = preprocess_image(img1, mtx, dist)
        warped2, _ = preprocess_image(img2, mtx, dist)

        y_mppx1, x_mppx1 = compute_mppx(warped1, dashed_line_loc='right')
        y_mppx2, x_mppx2 = compute_mppx(warped2, dashed_line_loc='left')

        # Finding average of pixel to meter in height and width
        x_mppx = (x_mppx1 + x_mppx2) / 2
        y_mppx = (y_mppx1 + y_mppx2) / 2

        # Save the computed calibration data
        with open(CAMERA_CALIB_PATH, mode='wb') as f:
            pickle.dump({'x_mppx': x_mppx, 'y_mppx': y_mppx, 'mtx': mtx, 'dist': dist}, f)

    return mtx, dist,x_mppx,y_mppx
 