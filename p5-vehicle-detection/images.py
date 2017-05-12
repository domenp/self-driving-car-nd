import os
import cv2
import numpy as np
from util import relevant_detections, draw_labeled_bboxes
from detector import VehicleDetector
from scipy.ndimage.measurements import label


def process_img(img, model):
    threshold = 1

    _, box_list = model.find_cars(img)
    _, box_list_scale_2 = model.find_cars(img, scale=2)
    box_list.extend(box_list_scale_2)
    heatmap = relevant_detections(img, box_list, threshold=threshold)

    return draw_labeled_bboxes(np.copy(img), label(heatmap))


if __name__ == '__main__':

    detector = VehicleDetector()
    detector.load('output/svc_pickle.p')

    path = 'udacity/test_images/'
    for img_name in os.listdir(path):
        if not img_name.endswith('jpg'):
            continue
        img = cv2.imread(path + img_name)
        result_img = process_img(img, detector)
        cv2.imwrite('test_images/' + img_name, result_img)
