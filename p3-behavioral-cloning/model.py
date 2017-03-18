import csv
import sys
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from PIL import Image


NB_EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 3
SHOW_IMAGE = False

IMAGE_NORMAL = 0
IMAGE_FLIPIT = 1

MODEL_COMMAAI = 'commaai'
MODEL_NVIDIA = 'nvidia'


# driving data for training
# format:
#   - path to directory,
#   - take only non-zero angles (True/False)
#   - take left and right camera images (True/False)
#   - flip the center image (True/False)
datasets = [
    # ('./data/udacity', False, True, True),
    ('./data/curves', False, True, True),
    # ('./data/wrong_way', False, False, True),
    ('./data/regular', False, True, True),
    ('./data/soft_left_turn', False, True, False),
    ('./data/sharp_left_curve', False, True, False),
    ('./data/sharp_turn_left_2', False, True, False),
    ('./data/sharp_left_turn_3', False, True, False),
    ('./data/sharp_left_turn_4', False, True, False),
    ('./data/sharp_left_turn_5', False, True, False),
    ('./data/sharp_left_turn_6', False, True, False),
    ('./data/sharp_right_turn', False, True, False),
    ('./data/sharp_turn_right_2', False, True, False),
    ('./data/sharp_right_turn_3', False, True, False),
    ('./data/sharp_right_turn_4', False, True, False),
    ('./data/sharp_right_turn_5', False, True, False),
    ('./data/sharp_right_turn_6', False, True, False),
    ('./data/sharp_right_turn_7', False, True, False)]


def load_all():
    """
    Load all available datasets.

    Calls `load_samples` to do the heavy lifting.
    """
    samples, angles = [], []
    for path, onza, tlr, aug in datasets:
        sd, ad = load_samples(path, only_non_zero_angles=onza, take_left_right=tlr, aug_center=aug)
        samples.extend(sd)
        angles.extend(ad)
    return samples, angles


def load_samples(base_path, only_non_zero_angles=False, take_left_right=False, aug_center=False):
    """
    Load a given dataset.

    Returns
    -------
    samples - list of tuples (image name, image type)
    angles - an angle corresponding to the sample image
    """
    samples, angles = [], []
    with open('%s/driving_log.csv' % base_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:

            center_image_name = get_image_path(row[0], base_path)
            center_angle = float(row[3])

            is_sharp_turn = center_angle > 0.025 or center_angle < -0.025

            if not only_non_zero_angles or is_sharp_turn:
                samples.append((center_image_name, IMAGE_NORMAL))
                angles.append(center_angle)

            if take_left_right:
                left_image_name = get_image_path(row[1], base_path)
                left_angle = center_angle + 0.2 if center_angle + 0.2 <= 1. else 1.
                samples.append((left_image_name, IMAGE_NORMAL))
                angles.append(left_angle)

                right_image_name = get_image_path(row[2], base_path)
                right_angle = center_angle - 0.2 if center_angle - 0.2 >= -1. else -1.
                samples.append((right_image_name, IMAGE_NORMAL))
                angles.append(right_angle)

            if aug_center:
                center_aug_image_name = get_image_path(row[0], base_path)
                center_aug_angle = -center_angle
                samples.append((center_aug_image_name, IMAGE_FLIPIT))
                angles.append(center_aug_angle)

    return samples, angles


def load_image(name, flipit=False, downscale=True):
    """
    Loads and crop an image.

    It flips the image in case `flipit` is set to True.

    Returns
    -------
    a numpy array representing the image
    """
    if downscale:
        load_size = (80, 160)
        crop_tuple = (0, 25, 160, 70)
    else:
        load_size = (160, 320)
        crop_tuple = (0, 50, 320, 140)

    img = load_img(name, target_size=load_size)
    img = img.crop(crop_tuple)
    if flipit:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if SHOW_IMAGE:
        img.save('output/%s_crop.jpg' % name.split('/')[-1])
    return img_to_array(img, dim_ordering='tf')


def get_image_path(name, base_path='data'):
    return '%s/IMG/%s' % (base_path, name.split('/')[-1])


def generator(samples, labels, downscale=True, batch_size=32):
    while 1:
        shuffle(samples, labels)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_angles = labels[offset:offset+batch_size]

            images = []
            angles = []
            for sample, angle in zip(batch_samples, batch_angles):
                if sample[1] == IMAGE_NORMAL:
                    img = load_image(sample[0], downscale=downscale)
                else:
                    img = load_image(sample[0], downscale=downscale, flipit=True)

                images.extend([img])
                angles.extend([angle])

            yield shuffle(np.array(images), np.array(angles))


def get_nvidia_model(learn_rate=0.001):
    """
    A model inspired by Nvidia's End to End learning paper.

    https://arxiv.org/abs/1604.07316
    """

    input_shape = IMG_SHAPE + (3,)

    model = Sequential()
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu", input_shape=input_shape))
    model.add(Convolution2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu"))

    model.add(Flatten())
    model.add(Dense(1164, activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))

    opt = Adam(lr=learn_rate)
    model.compile(optimizer=opt, loss="mse")

    return model


def get_commaai_model(learn_rate=0.001):
    """
    The code for the model was taken from
    https://github.com/commaai/research/blob/master/train_steering_model.py
    """
    input_shape = IMG_SHAPE + (3,)

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=input_shape,
                     output_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    opt = Adam(lr=learn_rate)
    model.compile(optimizer=opt, loss="mse")
    # print(model.summary())

    return model


def subsample_zero_angles(samples, angles, subsample_factor=5):
    """
    Reduce the number of zero angles (straight driving).
    """
    straight_driving = [(v, a) for v, a in zip(samples, angles) if a == 0.]
    turns = [(v, a) for v, a in zip(samples, angles) if a != 0.]
    straight_driving = shuffle(straight_driving)

    print('sample stats: turns %d straight driving %d' % (len(turns), len(straight_driving)))

    nb_straight = len(straight_driving) // subsample_factor
    turns.extend(straight_driving[:nb_straight])
    new_samples = [v[0] for v in turns]
    new_angles = [v[1] for v in turns]
    return new_samples, new_angles


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--quick', type=bool, default=False, help='quick run')
    parser.add_argument('--only-show-image', type=bool, default=False, help='show image')
    parser.add_argument('--early-stop', type=bool, default=True, help='early stop')
    parser.add_argument('--learn-rate', type=float, default=0.001, help='learn rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--subsample', type=bool, default=False, help='subsample zero angles')
    parser.add_argument('--model', type=str, default=MODEL_COMMAAI, help='type of a model to train')
    parser.add_argument('--output-model', type=str, default='model', help='name of the trained model')
    args = parser.parse_args()

    # load the driving data samples from all available datasets
    samples, angles = load_all()
    if args.subsample:
        samples, angles = subsample_zero_angles(samples, angles)

    if args.only_show_image:
        # only for debugging
        SHOW_IMAGE = True
        gen = generator(samples, angles, batch_size=32)
        batch, _ = next(gen)
        sys.exit(0)

    # shuffle the samples and split the data set into training and validation
    samples_train, samples_val, angles_train, angles_val = train_test_split(samples, angles, test_size=0.2, random_state=0)
    nb_samples_train, nb_samples_val = len(samples_train), len(samples_val)

    NB_EPOCHS = args.epochs
    if args.quick:
        nb_samples_train, nb_samples_val = 35, 32
        NB_EPOCHS = 1

    downscale = True
    model_path = 'models/%s.h5' % args.output_model

    if args.model == MODEL_COMMAAI:
        print('using commaai model')
        IMG_SHAPE = 45, 160
        model = get_commaai_model(args.learn_rate)
    elif args.model == MODEL_NVIDIA:
        print('using nvidia model')
        IMG_SHAPE = 90, 320
        downscale = False
        model = get_nvidia_model(args.learn_rate)
    else:
        sys.exit('no model specified, exiting')

    callbacks = []
    if args.early_stop:
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=1, mode='auto')
        callbacks.extend([checkpoint, early_stopping])

    print('training with learn rate: %f' % args.learn_rate)

    model.fit_generator(
      generator(samples_train, angles_train, downscale=downscale),
      samples_per_epoch=nb_samples_train,
      nb_epoch=NB_EPOCHS,
      validation_data=generator(samples_val, angles_val, downscale=downscale),
      nb_val_samples=nb_samples_val,
      callbacks=callbacks)

    if not args.early_stop:
        model.save(model_path)

    json.dump(datasets, open('models/%s.desc' % args.output_model, 'w'))
