# Advanced Lane Finding Project

The goal of this project was to create a robust pipeline (algorithm) to identify and mark a driving lane from a front facing camera.

An explanation of the pipeline is available in the writeup.

## Steps to reproduce the results

0. Camera calibration needs to be performed before any detection can take place. Calibration images are provided in the original repository. To perform the calibration run

`python camera_cal.py`  

1. Images for testing the pipeline are provided in the original repository. It's possible to reproduce the results by running

`python images.py`

This will save processed images (with the lane identified) into `output_images` directory.

2. The main output of the project is a video where a driving lane was identified and marked on every frame. It can be reproduced by running

`python video.py --input udacity/project_video.mp4 --output processed.mp4 --smooth=ma`
