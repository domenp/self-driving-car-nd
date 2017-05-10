import os
import cv2
import numpy as np
import pickle


CAMERA_CAL_PATH = 'camera_cal'
CAL_OUTPUT_DATA = 'data/calibration'


def calibrate_camera(debug=False):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for idx, fname in enumerate(os.listdir(CAMERA_CAL_PATH + '/')):
        if not fname.endswith('jpg'):
            continue
        img = cv2.imread(CAMERA_CAL_PATH + '/' + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if debug:
                cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                write_name = 'output_images/corners_found_' + fname
                cv2.imwrite(write_name, img)

    img = cv2.imread('camera_cal/calibration5.jpg')
    img_size = img.shape[:2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def get_calibration_data(debug=False):
    if not os.path.exists(CAL_OUTPUT_DATA):
        mtx, dist = calibrate_camera(debug=debug)
        pickle.dump((mtx, dist), open(CAL_OUTPUT_DATA, 'wb'))
    else:
        mtx, dist = pickle.load(open(CAL_OUTPUT_DATA, 'rb'))

    return mtx, dist
