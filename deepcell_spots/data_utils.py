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

"""Functions for making training data sets"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K


def slice_image(X, reshape_size, overlap=0):
    """Slice images in X into smaller parts.

    Similar to ``deepcell.utils.data_utils.reshape_matrix``.

    Args:
        X (np.array): array containing images with size
            `(img_number, y, x, channel)`.
        reshape_size (list): Shape of reshaped image `[y_size, x_size]`.
        overlap (int): Number of pixels overlapping in each row/column with
            the pixels from the same row/column in the neighboring slice.

    Returns:
        np.array: Stack of reshaped images in order of small to large y,
            then small to large x position in the original image
            np.array of size (n*img_number, y_size, x_size, channel)
            where n = number of images each image in X was sliced into
            if the original image lengths aren't divisible by y_size, x_size,
            the last image in each row / column overlaps with the one before.
    """
    image_size_x = X.shape[1]
    image_size_y = X.shape[2]

    L_x = reshape_size[0]  # x length of each slice
    L_y = reshape_size[1]  # y length of each slice

    # number of slices along x axis
    n_x = np.int(np.ceil((image_size_x - 2 * L_x + overlap) / (L_x - overlap)) + 2)
    # number of slices along y axis
    n_y = np.int(np.ceil((image_size_y - 2 * L_y + overlap) / (L_y - overlap)) + 2)

    new_batch_size = X.shape[0] * n_x * n_y  # number of images in output

    new_X_shape = (new_batch_size, L_x, L_y, X.shape[3])
    new_X = np.zeros(new_X_shape, dtype=K.floatx())

    counter = 0
    for b in range(X.shape[0]):
        for i in range(n_x):
            for j in range(n_y):
                _axis = 1
                if i != n_x - 1:
                    x_start, x_end = i * (L_x - overlap), i * \
                        (L_x - overlap) + L_x
                else:
                    x_start, x_end = -L_x, X.shape[_axis]

                if j != n_y - 1:
                    y_start, y_end = j * (L_y - overlap), j * \
                        (L_y - overlap) + L_y
                else:
                    y_start, y_end = -L_y, X.shape[_axis + 1]

                new_X[counter] = X[b, x_start:x_end, y_start:y_end, :]
                counter += 1

    print('Sliced data from {} to {}'.format(X.shape, new_X.shape))
    return new_X


def slice_annotated_image(X, y, reshape_size, overlap=0):
    """Slice images in X into smaller parts.

    Similar to ``deepcell.utils.data_utils.reshape_matrix``

    Args:
        X (np.array): array containing images with size
            `(img_number, y, x, channel)`.
        reshape_size (list): Shape of reshaped image `[y_size, x_size]`.
        overlap (int): Number of pixels overlapping in each row/column with
            the pixels from the same row/column in the neighboring slice.
        y: List or array containing coordinate annotations.
            Has length (img_number), each element of the list is a (N, 2)
            np.array where N=the number of points in the image.

    Returns:
        Two outputs (1) Stack of reshaped images in order of small to large y,
        then small to large x position in the original image np.array
        of size (n*img_number, y_size, x_size, channel) where n = number
        of images each image in X was sliced into if the original image
        lengths aren't divisible by y_size, x_size, the last image in
        each row / column overlaps with the one before and (2) list of
        length n*img_number
    """
    image_size_y = X.shape[1]
    image_size_x = X.shape[2]

    L_y = reshape_size[0]  # y length of each slice
    L_x = reshape_size[1]  # x length of each slice

    # number of slices along y axis
    n_y = np.int(
        np.ceil((image_size_y - 2 * L_y + overlap) / (L_y - overlap)) + 2)
    # number of slices along x axis
    n_x = np.int(
        np.ceil((image_size_x - 2 * L_x + overlap) / (L_x - overlap)) + 2)

    new_batch_size = X.shape[0] * n_y * n_x  # number of images in output

    new_X_shape = (new_batch_size, L_y, L_x, X.shape[3])
    new_X = np.zeros(new_X_shape, dtype=K.floatx())

    new_y = [None] * new_batch_size

    counter = 0
    for b in range(X.shape[0]):
        for i in range(n_y):
            for j in range(n_x):
                _axis = 1
                if i != n_y - 1:
                    y_start, y_end = i * (L_y - overlap), i * \
                        (L_y - overlap) + L_y
                else:
                    y_start, y_end = X.shape[_axis] - L_y, X.shape[_axis]

                if j != n_x - 1:
                    x_start, x_end = j * (L_x - overlap), j * \
                        (L_x - overlap) + L_x
                else:
                    x_start, x_end = X.shape[_axis + 1] - L_x, X.shape[_axis + 1]

                new_X[counter] = X[b, y_start:y_end, x_start:x_end, :]

                new_y[counter] = np.array(
                    [[y0 - y_start, x0 - x_start] for y0, x0 in y[b] if
                     (y_start - 0.5) <= y0 < (y_end - 0.5) and
                     (x_start - 0.5) <= x0 < (x_end - 0.5)])

                counter += 1

    print('Sliced data from {} to {}'.format(X.shape, new_X.shape))
    return new_X, new_y


def get_data(file_name, test_size=.2, seed=0, allow_pickle=False):
    """Load data from NPZ file and split into train and test sets.

    This is a copy of ``deepcell.utils.data_utils.get_data``,
    with `allow_pickle` added and `mode` removed.

    Args:
        file_name (str): path to NPZ file to load.
        test_size (float): percent of data to leave as testing holdout.
        seed: seed number for random train/test split repeatability.
        allow_pickle (bool): if True, allow loading pickled object arrays
            stored in npz files (via ``numpy.load``).

    Returns:
        dict: Dictionary of training data and a dictionary of testing data.
    """

    training_data = np.load(file_name, allow_pickle=allow_pickle)
    X = training_data['X']
    y = training_data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    train_dict = {
        'X': X_train,
        'y': y_train
    }

    test_dict = {
        'X': X_test,
        'y': y_test
    }

    return train_dict, test_dict
