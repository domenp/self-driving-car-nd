import cv2
import argparse
import numpy as np
from moviepy.editor import VideoFileClip
from util import relevant_detections, draw_labeled_bboxes, draw_heatmap_on_image
from detector import VehicleDetector
from collections import deque
from scipy.ndimage.measurements import label


class VideoProcessor(object):

    def __init__(self, model_path, threshold=1, debug=False):
        # number of detections needed to mark a blob in a histogram as a valid detection
        self.threshold = threshold
        # heatmaps from previous frames we keep to make pipeline more robust
        self.heatmaps = deque()

        self.debug = debug
        self.frames_processed = 0

        self.detector = VehicleDetector()
        self.detector.load(model_path)

    def process_frame(self, img):
        """
        Runs a sliding window search on every frame and returns positive vehicle detection.
        :param img:
        :return:
        """
        if self.debug:
            self.frames_processed += 1
            img_to_save = img.copy()
            cv2.imwrite('video_images/orig_frame_%d.jpg' %
                        self.frames_processed, cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB))

        _, box_list = self.detector.find_cars(img, src_color_space='RGB')
        _, box_list_scale2 = self.detector.find_cars(img, scale=2, src_color_space='RGB')
        box_list.extend(box_list_scale2)
        heatmap_frame = relevant_detections(img, box_list, threshold=self.threshold)
        if self.debug:
            img = draw_heatmap_on_image(img, heatmap_frame)

        # combine heatmaps from previous frames and threshold it to make pipeline more robust
        heatmap_combined = self.integrate_heatmaps(heatmap_frame)
        img = draw_labeled_bboxes(np.copy(img), label(heatmap_combined))

        if self.debug:
            img = draw_heatmap_on_image(img, heatmap_combined, left=450)
            cv2.imwrite('video_images/frame_%d.jpg' % self.frames_processed, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img

    def integrate_heatmaps(self, current, n_heatmaps=12, min_detections=7):
        """
        Combine n last frames' heatmaps together and threshold it to make the pipeline more robust.
        :param current: heatmap from the current frame
        :param n_heatmaps: number of heatmaps that we combine
        :param min_detections: a threshold of minimum positive detection to consider it as a valid detection
        :return: a heatmap
        """
        if len(self.heatmaps) < n_heatmaps-1:
            self.heatmaps.append(current)
            return current

        if len(self.heatmaps) > n_heatmaps:
            self.heatmaps.popleft()

        heatmap_combined = np.copy(current)
        for hm in self.heatmaps:
            heatmap_combined += hm
        heatmap_combined[heatmap_combined <= min_detections] = 0

        self.heatmaps.append(current)

        return heatmap_combined

    def process(self, src_path, dst_path):
        clip1 = VideoFileClip(src_path)
        lane_clip = clip1.fl_image(self.process_frame)
        lane_clip.write_videofile(dst_path, audio=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='udacity/project_video.mp4')
    parser.add_argument('--output', type=str, default='processed.mp4')
    parser.add_argument('--model-path', type=str, default='output/svc_pickle.p')
    parser.add_argument('--threshold', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    video = VideoProcessor(args.model_path, threshold=args.threshold, debug=args.debug)
    video.process(args.input, args.output)
