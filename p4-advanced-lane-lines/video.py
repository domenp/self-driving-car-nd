import argparse
import numpy as np
from camera_cal import get_calibration_data
from pipeline import *
from moviepy.editor import VideoFileClip


class VideoProcessor(object):

    def __init__(self, smooth = 'ma'):
        self.mtx, self.dist = get_calibration_data(debug=False)
        self.left_fit, self.right_fit = (), ()
        self.left_fit_hist = []
        self.right_fit_hist = []
        self.curvature_hist = []
        self.smooth = smooth
        self.prev_left_fit = ()
        self.prev_right_fit = ()

    def moving_average(self, n=15):

        left_fit_adjust = np.average(self.left_fit_hist[-n:], axis=0)
        right_fit_adjust = np.average(self.right_fit_hist[-n:], axis=0)

        return left_fit_adjust, right_fit_adjust

    def exp_smoothing(self, alpha=0.1):
        if not self.prev_left_fit or not self.prev_right_fit:
            return self.left_fit, self.right_fit
        left_adjust = alpha * np.array(self.left_fit) + (1. - alpha) * np.array(self.prev_left_fit)
        right_adjust = alpha * np.array(self.right_fit) + (1. - alpha) * np.array(self.prev_right_fit)

        self.prev_left_fit = self.left_fit
        self.prev_right_fit = self.right_fit

        return left_adjust, right_adjust

    def any_outlier(self):
        if len(self.left_fit_hist) > 15 and len(self.right_fit_hist) > 15:
            return is_outlier(self.left_fit_hist, self.left_fit) or is_outlier(self.right_fit_hist, self.right_fit)
        return False

    def process_frame(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        img_bin = binarize(undist)
        warped = warp(img_bin)
        if len(self.left_fit) > 0 and len(self.right_fit) > 0:
            self.left_fit, self.right_fit = find_lane_fast(warped, self.left_fit, self.right_fit)
        else:
            self.left_fit, self.right_fit = find_lane_slow(warped)

        if not self.any_outlier():

            self.left_fit_hist.append(self.left_fit)
            self.right_fit_hist.append(self.right_fit)

            if self.smooth == 'ma':
                left_fit_adjust, right_fit_adjust = self.moving_average()
            elif self.smooth == 'exp':
                left_fit_adjust, right_fit_adjust = self.exp_smoothing()
            else:
                left_fit_adjust, right_fit_adjust = self.left_fit, self.right_fit

            self.prev_left_fit = left_fit_adjust
            self.prev_right_fit = right_fit_adjust
        else:
            left_fit_adjust, right_fit_adjust = self.prev_left_fit, self.prev_right_fit

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit_adjust[0] * ploty ** 2 + left_fit_adjust[1] * ploty + left_fit_adjust[2]
        right_fitx = right_fit_adjust[0] * ploty ** 2 + right_fit_adjust[1] * ploty + right_fit_adjust[2]

        result = draw_lane(warped, undist, ploty, left_fitx, right_fitx)

        curvature = compute_curvature(ploty, left_fitx, right_fitx)
        self.curvature_hist.append(curvature)
        curvature_adjust = np.average(self.curvature_hist[-15:], axis=0)
        offset = measure_offset(warped.shape[1] / 2, left_fitx, right_fitx)

        result = draw_text(result, 'radius of curvature: %dm' % int(curvature_adjust), (450, 50))
        result = draw_text(result, 'offset from center: %.2fm' % offset, (450, 100))

        return result

    def process(self, src_path, dst_path):
        clip1 = VideoFileClip(src_path)
        lane_clip = clip1.fl_image(self.process_frame)
        lane_clip.write_videofile(dst_path, audio=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='udacity/project_video.mp4')
    parser.add_argument('--output', type=str, default='processed.mp4')
    parser.add_argument('--smooth', type=str, default='no')
    args = parser.parse_args()

    float_formatter = lambda x: "%.5f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})

    video = VideoProcessor(smooth=args.smooth)
    video.process(args.input, args.output)