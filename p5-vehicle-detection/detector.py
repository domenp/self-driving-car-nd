"""
Vehicle detection.

Some of the code is taken or is inspired by the code from Udacity's SDC lessons.
"""
import os
import cv2
import pickle
import time
import argparse
import numpy as np
from util import get_hog_features, bin_spatial, color_hist, convert_color, get_image_paths
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


class VehicleDetector(object):
    """
    Trains a vehicle detector and provides methods for detections.
    """

    def __init__(self):
        # vehicle detector
        self.svc = None
        # transforms the dataset to zero mean unit variance
        self.X_scaler = None
        # number of orientation (HOG features)
        self.orient = 18
        # pixels per cell (HOG features)
        self.pix_per_cell = 8
        # cells per block (HOG features)
        self.cell_per_block = 2
        # size of spatially binned color histogram
        self.spatial_size = (32, 32)
        # number of color histogram bins
        self.hist_bins = 32
        # color space to use for feature extract
        self.colorspace = 'LUV'
        # search area top y coordinate
        self.ystart = 400
        # search area bottom y coordinate
        self.ystop = 656

    def train(self, force_feature_extract=False):
        """
        Train the vehicle detector.

        Splits the dataset into training and testing portion and extracts image features for both of them. Trains
        the SVM classifier using HOG features combined with spatially binned color histogram and a color histogram.

        :param force_feature_extract: extract train features from the image if true else it uses the cached ones
        :return:
        """
        cars_base_path = 'data/vehicles/'
        notcars_base_path = 'data/non-vehicles/'

        cars_gti_paths = get_image_paths(cars_base_path, ['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right'])
        cars_kitti_paths = shuffle(get_image_paths(cars_base_path, ['KITTI_extracted']))
        notcars_gti_paths = get_image_paths(notcars_base_path, ['GTI'])
        notcars_extras_paths = shuffle(get_image_paths(notcars_base_path, ['Extras']))

        n_test_cars = min(len(cars_kitti_paths), int((len(cars_gti_paths) + len(cars_kitti_paths)) * 0.2))
        n_test_notcars = min(len(notcars_extras_paths), int((len(notcars_gti_paths) + len(notcars_extras_paths)) * 0.2))

        cars_train_paths = cars_gti_paths
        cars_train_paths.extend(cars_kitti_paths[:-n_test_cars])
        cars_test_paths = cars_kitti_paths[-n_test_cars:]

        notcars_train_paths = notcars_gti_paths
        notcars_train_paths.extend(notcars_extras_paths[:-n_test_notcars])
        notcars_test_paths = notcars_extras_paths[-n_test_notcars:]

        print('computing features')

        t = time.time()
        car_train_features = self.extract_train_features(cars_train_paths, 'output/car_train_features.p', force_feature_extract)
        car_test_features = self.extract_train_features(cars_test_paths, 'output/car_test_features.p', force_feature_extract)
        notcar_train_features = self.extract_train_features(notcars_train_paths, 'output/notcar_train_features.p', force_feature_extract)
        notcar_test_features = self.extract_train_features(notcars_test_paths, 'output/notcar_test_features.p', force_feature_extract)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')

        print('got %d training samples: %d cars, %d non cars' %
              (len(car_train_features) + len(notcar_train_features), len(car_train_features), len(notcar_train_features)))

        # Create an array stack of feature vectors
        X_train = np.vstack((car_train_features, notcar_train_features)).astype(np.float64)
        X_test = np.vstack((car_test_features, notcar_test_features)).astype(np.float64)

        self.X_scaler = StandardScaler().fit(X_train)
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        y_train = np.hstack((np.ones(len(car_train_features)), np.zeros(len(notcar_train_features))))
        y_test = np.hstack((np.ones(len(car_test_features)), np.zeros(len(notcar_test_features))))

        print('Using:', self.colorspace, 'color space, ',  self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        t = time.time()
        self.svc = LinearSVC()
        self.svc.fit(X_train, y_train)
        t2 = time.time()

        print(round(t2 - t, 2), 'Seconds to train SVC...')
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))

    def extract_train_features(self, image_paths, cache_path, force_feature_extract=False):
        """
        Extract features from images provided as image paths.
        :param image_paths: list of image paths to extract the features from
        :param cache_path: path to the image feature cache
        :param force_feature_extract: force recomputation of image features if true
        :return:
        """
        features = []
        if not os.path.exists(cache_path) or force_feature_extract:
            for img_path in image_paths:
                img = cv2.imread(img_path)
                img = convert_color(img, dst_space=self.colorspace)
                hog_feat1 = get_hog_features(img[:, :, 0], self.orient, self.pix_per_cell, self.cell_per_block,
                                             vis=False, feature_vec=True).ravel()
                hog_feat2 = get_hog_features(img[:, :, 1], self.orient, self.pix_per_cell, self.cell_per_block,
                                             vis=False, feature_vec=True).ravel()
                hog_feat3 = get_hog_features(img[:, :, 2], self.orient, self.pix_per_cell, self.cell_per_block,
                                             vis=False, feature_vec=True).ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                features.append(self.feature_vector(img, hog_features))

            pickle.dump(features, open(cache_path, 'wb'))
        else:
            features = pickle.load(open(cache_path, 'rb'))

        return features

    def feature_vector(self, img, hog_features):
        """
        Stack features in the final feature vector.
        :param img:
        :param hog_features:
        :return:
        """
        spatial_features = bin_spatial(img, size=self.spatial_size)
        hist_features = color_hist(img, nbins=self.hist_bins)
        return np.hstack((spatial_features, hist_features, hog_features))

    def find_cars(self, img, src_color_space='BGR', scale=1):
        """
        Run the sliding window over an images and return the position of positive detections.
        :param img: source image
        :param src_color_space: color space of the source image
        :param scale: used to rescale the original image
        :return: an image overlaid by bouding boxes of positive detection, list of bounding box coordinates
        """
        draw_img = np.copy(img)

        img_search = img[self.ystart:self.ystop, :, :]
        img_search = convert_color(img_search, src_space=src_color_space, dst_space=self.colorspace)
        if scale != 1:
            imshape = img_search.shape
            img_search = cv2.resize(img_search, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = img_search[:, :, 0]
        ch2 = img_search[:, :, 1]
        ch3 = img_search[:, :, 2]

        nxcells = (ch1.shape[1] // self.pix_per_cell) - 1
        nycells = (ch1.shape[0] // self.pix_per_cell) - 1

        window_size = 64
        ncells_per_window = (window_size // self.pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxcells - ncells_per_window) // cells_per_step
        nysteps = (nycells - ncells_per_window) // cells_per_step

        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        box_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                hog_feat1 = hog1[ypos:ypos + ncells_per_window, xpos:xpos + ncells_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + ncells_per_window, xpos:xpos + ncells_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + ncells_per_window, xpos:xpos + ncells_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_search[ytop:ytop + window_size, xleft:xleft + window_size], (64, 64))

                # Scale features and make a prediction
                combined = self.feature_vector(subimg, hog_features)
                test_features = self.X_scaler.transform(combined.reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window_size * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + self.ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + self.ystart), (0, 0, 255), 6)
                    box_list.append(
                        [[xbox_left, ytop_draw + self.ystart],
                         [xbox_left + win_draw, ytop_draw + win_draw + self.ystart]])

        return draw_img, box_list

    def load(self, model_path):
        """
        Load the detector from cache.
        """
        data = pickle.load(open(model_path, "rb"))
        self.svc = data['svc']
        self.X_scaler = data['scaler']

    def save(self, model_path):
        """
        Save the detector for later use.
        """
        pickle.dump({'svc': self.svc, 'scaler': self.X_scaler}, open(model_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-feature-extract', type=bool, default=False)
    args = parser.parse_args()

    model = VehicleDetector()
    model.train(force_feature_extract=args.force_feature_extract)
    model.save('output/svc_pickle.p')
