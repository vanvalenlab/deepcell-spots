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

"""Spot detection image generators"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  Iterator,
                                                  array_to_img)

from deepcell_spots.utils import (affine_transform_points,
                                  subpixel_distance_transform)


class ImageFullyConvDotIterator(Iterator):
    """Iterator yielding data from Numpy arrays (`X` and `y`).

    Args:
        train_dict (dict): dictionary consisting of numpy arrays for `X` and `y`.
            X has dimensions (batch, Ly, Lx, channel) and y has length batch
            containing np.arrays of shape (N, 2) where N is the number of
            points in the image.
        image_data_generator: Instance of `ImageDataGenerator` to use for random
            transformations and normalization.
        batch_size (int): Size of a batch.
        shuffle (bool): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        data_format (str): One of (`channels_first`, `channels_last`).
        save_to_dir (str): Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful for visualizing
            the random transformations being applied, for debugging purposes.
        save_prefix (str): Prefix to use for saving sample images (if `save_to_dir`
            is set).
        save_format (str): Format to use for saving sample images (if `save_to_dir`
            is set).

    Raises:
        ValueError: Training batches and labels should have the same length.
        ValueError: Input data in `ImageFullyConvIterator` should have rank 4.
    """

    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 skip=None,
                 shuffle=False,
                 transform=None,
                 transform_kwargs={},
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):

        X, y = train_dict['X'], train_dict['y']
        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))
        self.x = np.asarray(X, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageFullyConvIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.y = y
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImageFullyConvDotIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def point_list_to_annotations(self, points, image_shape, dy=1, dx=1):
        """ Generate label images used in loss calculation from point labels.

        Args:
            points (np.array): array of size (N, 2) which contains points in the format [y, x].
            image_shape (tuple): shape of 2-dimensional image.
            dy: pixel y width.
            dx: pixel x width.

        Returns:
            annotations (dict): Dictionary with two keys, `detections` and `offset`.
                - `detections` is array of shape (image_shape,2) with pixels one hot encoding
                  spot locations.
                - `offset` is array of shape (image_shape,2) with pixel values equal to
                  signed distance to nearest spot in x- and y-directions.
        """

        contains_point = np.zeros(image_shape)
        for ind, [y, x] in enumerate(points):
            nearest_pixel_x_ind = int(round(x / dx))
            nearest_pixel_y_ind = int(round(y / dy))
            contains_point[nearest_pixel_y_ind, nearest_pixel_x_ind] = 1

        delta_y, delta_x, _ = subpixel_distance_transform(
            points, image_shape, dy=1, dx=1)
        offset = np.stack((delta_y, delta_x), axis=-1)

        one_hot_encoded_cp = to_categorical(contains_point)

        # TODO: refactor detections to classification
        annotations = {'detections': one_hot_encoded_cp, 'offset': offset}
        return annotations

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))

        batch_detections = []  # eventual shape: (batch, Lx, Ly, 2)
        batch_offset = []  # eventual shape: (batch, Lx, Ly, 2)

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(
                    x.astype(K.floatx()), y)

                annotations = self.point_list_to_annotations(y, x.shape[:2])

                batch_detections.append(annotations['detections'])
                batch_offset.append(annotations['offset'])
            else:
                x = self.image_data_generator.random_transform(
                    x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                if self.data_format == 'channels_first':
                    img_x = np.expand_dims(batch_x[i, 0, ...], 0)
                else:
                    img_x = np.expand_dims(batch_x[i, ..., 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                # save the 3 images of labels for each x image
                if self.y is not None:
                    # Save detections image
                    # get first channel of i-th image in batch
                    img_y = np.expand_dims(batch_detections[i, ..., 0], -1)
                    img_y = np.expand_dims(img_y, axis=self.channel_axis - 1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_det_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

                    # save offset images
                    # y offset, first channel, i-th image of batch
                    img_y = np.expand_dims(batch_offset[i, ..., 0, 0], -1)
                    img_y = np.expand_dims(img_y, axis=self.channel_axis - 1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_y_offset_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

                    img_y = np.expand_dims(batch_offset[i, ..., 0, 1], -1)
                    img_y = np.expand_dims(img_y, axis=self.channel_axis - 1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_x_offset_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x

        batch_detections = np.stack(batch_detections, axis=0)
        batch_offset = np.stack(batch_offset, axis=0)
        batch_y = [batch_offset, batch_detections]

        return batch_x, batch_y

    def next(self):
        """For python 2.x. Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class ImageFullyConvDotDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.

    The data will be looped over in batches.

    Args:
        featurewise_center (bool): Whether to set input mean to 0 over the
            dataset, feature-wise.
        samplewise_center (bool): Whether to set each sample mean to 0.
        featurewise_std_normalization (bool): Whether to divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization (bool): Whether to divide each input by
            its std.
        zca_epsilon (float): Epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening (bool): Whether to apply ZCA whitening.
        rotation_range (int): Degree range for random rotations.
        width_shift_range: float, 1-D array-like or int.
            Values for the following formats:
                - float: fraction of total width, if < 1, or pixels if >= 1.
                - 1-D array-like: random elements from the array.
                - int: integer number of pixels from interval
                  `(-width_shift_range, +width_shift_range)` with
                  `width_shift_range=2` possible values are ints [-1, 0, +1],
                  same as with `width_shift_range=[-1, 0, +1]`, while with
                  `width_shift_range=1.0` possible values are floats in
                  the interval [-1.0, +1.0).
        shear_range (float): Shear angle in counter-clockwise direction in
            degrees
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range (float): Range for random channel shifts.
        fill_mode (str): One of `("constant", "nearest", "reflect" or "wrap")`.
            Default is `"nearest"`. Points outside the boundaries of the input
            are filled according to the given mode:
                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd
        cval: float or int, value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip (bool): Whether to randomly flip inputs horizontally.
        vertical_flip (bool): Whether to randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one image (Numpy tensor with rank 3) as
            argument and should output a Numpy tensor with the same shape.
        data_format: One of `("channels_first", "channels_last")`.
            `"channels_last"` means that the images should have shape
            `(samples, height, width, channels)`, `"channels_first"`
            means that the images should have shape
            `(samples, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        validation_split (float): Fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def flow(self,
             train_dict,
             batch_size=1,
             skip=None,
             transform=None,
             transform_kwargs={},
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict (dict): dictionary of X and y tensors. Both should be rank 4.
            batch_size (int): Size of a batch.
            shuffle (bool): Whether to shuffle the data between epochs.
            seed (int): Random seed for data shuffling.
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful for visualizing
                the random transformations being applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample images (if `save_to_dir`
                is set).
            save_format (str): Format to use for saving sample images (if `save_to_dir`
                is set).

        Returns:
            An Iterator yielding tuples of `(x, y)` where `x` is a numpy array
            of image data and `y` is a numpy array of labels of the same shape.
        """
        return ImageFullyConvDotIterator(
            train_dict,
            self,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=transform_kwargs,
            skip=skip,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def apply_points_transform(self, y, transform_parameters, image_shape):
        """Applies an affine transformation to a list of point coordinates according to
        given parameters.

        Args:
            y (np.array): Array of shape (N, 2) which contains points in the format [y, x]
                or list of such arrays. (y Cartesian coordinate before the x, as in
                matrix/image indexing convention. Not to be confused with the variables X,y
                as in data and labels)
            transform_parameters (dict): Dictionary with string - parameter pairs
                describing the transformation.
                    - `'theta'`: Float. Rotation angle in degrees.
                    - `'tx'`: Float. Shift in the x direction.
                    - `'ty'`: Float. Shift in the y direction.
                    - `'shear'`: Float. Shear angle in degrees.
                    - `'zx'`: Float. Zoom in the x direction.
                    - `'zy'`: Float. Zoom in the y direction.
                    - `'flip_horizontal'`: Boolean. Horizontal flip.
                    - `'flip_vertical'`: Boolean. Vertical flip.
                    - `'channel_shift_intensity'`: Float. Channel shift intensity.
                    - `'brightness'`: Float. Brightness shift intensity.
                      (taken from: keras ImageDataGenerator documentation)
            image_shape (tuple): shape of 2-dimensional image.
        """

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        y = affine_transform_points(y, transform_parameters,
                                    image_shape=image_shape,
                                    img_row_axis=img_row_axis,
                                    img_col_axis=img_col_axis,
                                    fill_mode=self.fill_mode)

        return y

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image

        Args:
            x: 3D tensor or list of 3D tensors.
            y (np.array): Array of shape (N, 2) which contains points in the format [y, x]
                or list of such arrays referring to coordinates in the image `x`, optional.
            seed: Random seed.
            fill_mode (str): One of `("constant", "nearest", "reflect" or "wrap")` for type
                of padding used for points outside of the input image which correspond to
                points inside the output image. Points outside the boundaries of the input
                are filled according to the given mode:
                    - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                    - 'nearest':  aaaaaaaa|abcd|dddddddd
                    - 'reflect':  abcddcba|abcd|dcbaabcd
                    - 'wrap':  abcdabcd|abcd|abcdabcd

        Returns:
            A randomly transformed version of the input (same shape). If `y` is passed, it is
            transformed if necessary and returned. The transformed y contains input and padding
            (for fill_mode='reflect' or 'wrap') points mapped to output image space, which are
            inside the output image (transformed points mapped to outside of the output image
            boundaries are deleted)
        """
        params = self.get_random_transform(x.shape, seed)

        # TODO
        # channel support code below (?) commented out for now
        # if isinstance(x, list):
        #    x = [self.apply_transform(x_i, params) for x_i in x]
        # else:
        x = self.apply_transform(x, params)

        if y is None:
            return x

        # apply the transform to the point labels
        # channel support code below (?) commented out for now
        # if isinstance(y, list):
        #    y = [self.apply_points_transform(y_i, params, image_shape=x.shape[:2]) for y_i in y]
        # else:
        y = self.apply_points_transform(y, params, image_shape=x.shape[:2])

        return x, y
