import os
from camera_cal import get_calibration_data
from pipeline import *


def process_single_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_bin = binarize(undist)
    warped = warp(img_bin)

    left_fit, right_fit = find_lane_slow(warped)
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return draw_lane(warped, undist, ploty, left_fitx, right_fitx)


def process_test_images():
    mtx, dist = get_calibration_data(debug=True)

    for name in os.listdir(TEST_IMAGES_DIR):
        if not name.endswith('jpg'):
            continue
        img = cv2.imread(TEST_IMAGES_DIR + '/' + name)
        result = process_single_image(img, mtx, dist)
        cv2.imwrite(OUTPUT_DIR + '/' + name, result)


if __name__ == '__main__':
    process_test_images()
