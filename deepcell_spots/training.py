# Copyright 2019-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-spots/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for training convolutional neural networks"""

from __future__ import absolute_import, division, print_function

import datetime
import os

import numpy as np
# import deepcell.losses
from deepcell.utils import train_utils
from deepcell.utils.train_utils import rate_scheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

from deepcell_spots import dotnet_losses, image_generators
from deepcell_spots.data_utils import get_data


def train_model_dot(model,
                    dataset,
                    expt='',
                    test_size=.2,
                    seed=0,
                    n_epoch=10,
                    batch_size=1,
                    num_gpus=None,
                    frames_per_batch=5,
                    optimizer=SGD(lr=0.01, decay=1e-6,
                                  momentum=0.9, nesterov=True),
                    log_dir='/data/tensorboard_logs',
                    model_dir='/data/models',
                    model_name=None,
                    focal=False,
                    sigma=3.0,
                    alpha=0.25,
                    gamma=0.5,
                    lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                    rotation_range=0,
                    flip=True,
                    shear=0,
                    zoom_range=0,
                    fill_mode='nearest',
                    cval=0.,
                    **kwargs):
    """Train a dot center detection model using fully convolutional mode.
        Args:
            model (tensorflow.keras.Model): The model to train.
            dataset (str): Path to a dataset to train the model with.
            expt (str): Experiment, substring to include in model name.
            test_size (float): Percent of data to leave as test data.
            seed (int): Random seed used for train-test split
            n_epoch (int): Number of training epochs.
            batch_size (int): Number of batches per training step.
            num_gpus (int): The number of GPUs to train on.
            frames_per_batch (int): Number of training frames if training 3D data.
            log_dir (str): Filepath to save tensorboard logs. If None, disables
                the tensorboard callback.
            model_dir (str): Directory to save the model file.
            model_name (str): Name of the model (and name of output file).
            focal (bool): If true, uses focal loss.
            sigma (float): The point where the loss changes from L2 to L1.
            alpha (float): Scale the focal weight with alpha.
            gamma (float): Parameter for focal loss (Take the power of the focal weight with gamma.)
            optimizer (object): Pre-initialized optimizer object (SGD, Adam, etc.)
            lr_sched (function): Learning rate scheduler function
            rotation_range (int): Maximum rotation range for image augmentation
            flip (bool): Enables horizontal and vertical flipping for augmentation
            shear (int): Maximum shear range for image augmentation
            zoom_range (tuple): Minimum and maximum zoom values (0.8, 1.2)
            fill_mode (str): padding style for data augmentation (input parameter of
                            tf.keras.preprocessing.image.ImageDataGenerator)
            cval (float or int): used for pixels outside the boundaries of the input image when
                            fill_mode='constant'
            kwargs (dict): Other parameters to pass to _transform_masks
        Returns:
            tensorflow.keras.Model: The trained model
    """

    is_channels_first = K.image_data_format() == 'channels_first'

    if model_name is None:
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = '{}_{}_{}'.format(todays_date, data_name, expt)
    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))

    train_dict, test_dict = get_data(
        dataset, test_size=test_size, seed=seed, allow_pickle=True)

    n_classes = model.layers[-1].output_shape[1 if is_channels_first else -1]
    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    if num_gpus >= 2:
        batch_size = batch_size * num_gpus
        model = train_utils.MultiGpuModel(model, num_gpus)

    print('Training on {} GPUs'.format(num_gpus))

    losses = dotnet_losses.DotNetLosses(
        sigma=sigma, alpha=alpha, gamma=gamma, focal=focal)

    loss = {
        'offset_regression': losses.regression_loss,
        'classification': losses.classification_loss
    }

    loss_weights = {"offset_regression": 1.0, "classification": 1.0}
    model.compile(loss=loss, loss_weights=loss_weights,
                  optimizer=optimizer, metrics=['accuracy'])

    if num_gpus >= 2:
        # Each GPU must have at least one validation example
        if test_dict['y'].shape[0] < num_gpus:
            raise ValueError('Not enough validation data for {} GPUs. '
                             'Received {} validation sample.'.format(
                                 test_dict['y'].shape[0], num_gpus))

        # When using multiple GPUs and skip_connections,
        # the training data must be evenly distributed across all GPUs
        num_train = train_dict['y'].shape[0]
        nb_samples = num_train - num_train % batch_size
        if nb_samples:
            train_dict['y'] = train_dict['y'][:nb_samples]
            train_dict['X'] = train_dict['X'][:nb_samples]

    # this will do preprocessing and realtime data augmentation
    # create DataGenerator object for generating augmented training data
    datagen = image_generators.ImageFullyConvDotDataGenerator(
        rotation_range=rotation_range,
        shear_range=shear,
        zoom_range=zoom_range,
        horizontal_flip=flip,
        vertical_flip=flip,
        fill_mode=fill_mode,
        cval=cval)

    # DataGenerator object for validation data - generates data with no augmentation
    datagen_val = image_generators.ImageFullyConvDotDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    # TO DO: add 3D support or delete redundant ndim==5 part
    if train_dict['X'].ndim == 5:
        train_data = datagen.flow(
            train_dict,
            # skip=skip,
            seed=seed,
            batch_size=batch_size,
            frames_per_batch=frames_per_batch)

        val_data = datagen_val.flow(
            test_dict,
            # skip=skip,
            seed=seed,
            batch_size=batch_size,
            frames_per_batch=frames_per_batch)
    else:
        train_data = datagen.flow(
            train_dict,
            # skip=skip,
            seed=seed,
            batch_size=batch_size)

        val_data = datagen_val.flow(
            test_dict,
            # skip=skip,
            seed=seed,
            batch_size=batch_size)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=[
            callbacks.LearningRateScheduler(lr_sched),
            callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=num_gpus >= 2),
            callbacks.TensorBoard(log_dir=os.path.join(log_dir, model_name))
        ])

    model.save_weights(model_path)
    np.savez(loss_path, loss_history=loss_history.history)

    return model
