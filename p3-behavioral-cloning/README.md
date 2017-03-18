# Behavioral cloning project

The goal of the project was to create a deep learning model that is able to steer a car in a simulator. It is supposed to achieve that by cloning driving behavior of a human driver. In our case the driving data came from simulator instead. The input to our model are images from a front facing camera together with a current steering angle of a car. More detailed description can be found in [the Udacity project repo](https://github.com/udacity/CarND-Behavioral-Cloning-P3).

The architecture used to build the model is inspired by [Nvidia End to End Learning paper](https://arxiv.org/abs/1604.07316). More detailed description how the model was trained is available in [the writeup](https://github.com/domenp/self-driving-car-nd/blob/master/p3-behavioral-cloning/report/writeup_report.md).

The final model was trained running the following command:

`python model.py --subsample=1 --model=nvidia --learn-rate=0.0001 --output-model=model`

Note: Training data is not provided in the repo due to its size. It is possible to obtain it by running [the Udacity's self-driving simulator](https://github.com/udacity/self-driving-car-sim).

Steer the car in the simulator:

`python drive.py models/model.h5`
