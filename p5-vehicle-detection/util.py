"""
Utility function for vehicle detection.

Some of the code is taken or is inspired by the code from Udacity's SDC lessons.
"""
import os
import cv2
import numpy as np
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def convert_color(img, src_space='BGR', dst_space='LUV'):
    if src_space == 'BGR':
        if dst_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif dst_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif dst_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif src_space == 'RGB':
        if dst_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif dst_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif dst_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        else:
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def bin_spatial(img, size=(32, 32)):
    """
    Return spatially binned color histogram.
    :param img: the original image
    :param size: the size of the image for computing the histogram
    :return: numpy array containing 3-channel color histogram
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    """
    Return color histogram.
    :param img: the original image
    :param nbins: the number of bins for the histogram
    :return: numpy array of 3-channel color histogram
    """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def relevant_detections(img, box_list, threshold=1):
    """
    Return relevant (valid) detections given the list of candidates.
    :param img: the original image
    :param box_list: list of candidates
    :param threshold: minimum number of detections to call a heatmap blob a valid detection
    :return: thresholded heatmap
    """
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    for box in box_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heat[heat <= threshold] = 0
    heatmap = np.clip(heat, 0, 255)

    return heatmap


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img


def draw_heatmap_on_image(img, heatmap, top=0, left=0):
    heatmap_small = cv2.resize(heatmap, (427, 240))
    heatmap_small = (heatmap_small / np.max(heatmap_small)) * 255.
    heatmap_small = np.dstack((heatmap_small, heatmap_small, heatmap_small))
    img[top:240+top, left:427+left] = heatmap_small
    return img


def get_image_paths(base_path, dirs):
    return [base_path + img_dir + '/' + v for img_dir in dirs
                for v in os.listdir(base_path + img_dir) if v.endswith('png')]