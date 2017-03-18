# Behavioral cloning

A trained model that steers a car in a simulator from Udacity. The model is inspired by Nvidia End to End Learning paper.

It was trained for 10 epochs with learning rate 0.0001. The dataset was comprised from 60k samples.

Apart from regular driving some additional data driving the sharp left and sharp right turn was provided. Special care was given to balance the dataset i.e. subsample straight driving samples (zero angle). 
