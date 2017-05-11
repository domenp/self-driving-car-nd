# Advanced Lane Finding Project

The goal of this project was to create a robust pipeline (algorithm) to identify the lane boundaries in a video.

An explanation of the pipeline is available in the [writeup](writeup.md).

## Steps to reproduce the results

0. Camera calibration needs to be performed before any lane detection can take place.

`python camera_cal.py`

This will perform the calibration using the calibration images from [the original repository](https://github.com/udacity/CarND-Advanced-Lane-Lines). The obtained parameters will be cached and saved as a `calibration` file in `data` directory.

1. Images for testing the pipeline are provided in [the original repository](https://github.com/udacity/CarND-Advanced-Lane-Lines). The processed images (with the lane boundaries marked) are available in `output_images` directory. It's possible to reproduce the results by running

`python images.py`

This will overwrite the images already present in `output_images` directory.

2. The main output of the project is a video where a driving lane was identified and marked on every frame. The video is available [here](https://youtu.be/rTAJ8oTjSGk). It can be reproduced by running

`python video.py --input udacity/project_video.mp4 --output processed.mp4 --smooth=ma`
