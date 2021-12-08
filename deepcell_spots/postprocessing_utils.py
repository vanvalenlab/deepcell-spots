# Copyright 2019-2021 The Van Valen Lab at the California Institute of
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

"""Functions that convert deep learning model output to list of detected spots"""

import numpy as np
from skimage import measure
from skimage.feature import peak_local_max


def y_annotations_to_point_list(y_pred, threshold):
    """Convert raw prediction to a predicted point list: classification of pixel as containing dot
    > threshold

    Args:
    y_pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for
        image #ind in the batch where annot_type = 0 or 1: 0 - contains_dot, 1 - offset matrices

    ind: the index of the image in the batch for which to convert the annotations

    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered
        containing a spot center, and their corresponding regression values will be used to create a
        final spot position prediction which will be added to the output spot center coordinates
        list.

    Returns:
    A list of spot center coordinates of the format [[y0, x0], [y1, x1],...]
    """
    dot_centers = []
    for ind in range(np.shape(y_pred)[1]):
        contains_dot = y_pred[1][ind, ..., 1] > threshold
        delta_y = y_pred[0][ind, ..., 0]
        delta_x = y_pred[0][ind, ..., 1]

        dot_pixel_inds = np.argwhere(contains_dot)
        dot_centers.append([[y_ind + delta_y[y_ind, x_ind], x_ind +
                             delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])

    return np.array(dot_centers)


def y_annotations_to_point_list_restrictive(y_pred, threshold):
    """ Convert raw prediction to a predicted point list: classification of pixel as containing dot
    > threshold AND center regression is contained in the pixel

    Args:
    y_pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for
    image #ind in the batch where annot_type = 0 or 1: 0 - contains_dot, 1 - offset matrices

    ind: the index of the image in the batch for which to convert the annotations

    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered
    containing a spot center, and their corresponding regression values will be used to create a
    final spot position prediction which willbe added to the output spot center coordinates list.

    Returns:
    A list of spot center coordinates of the format [[y0, x0], [y1, x1],...]
    """
    dot_centers = []
    for ind in range(np.shape(y_pred)[1]):
        contains_dot = y_pred[1][ind, ..., 1] > threshold
        delta_y = y_pred[0][ind, ..., 0]
        delta_x = y_pred[0][ind, ..., 1]
        contains_its_regression = (abs(delta_x) <= 0.5) & (abs(delta_y) <= 0.5)

        final_dot_detection = contains_dot & contains_its_regression

        dot_pixel_inds = np.argwhere(final_dot_detection)
        dot_centers.append(np.array(
            [[y_ind + delta_y[y_ind, x_ind],
              x_ind + delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds]))

    return np.array(dot_centers)


def y_annotations_to_point_list_max(y_pred, threshold=0.95, min_distance=2):
    """ Convert raw prediction to a predicted point list using PLM to determine local maxima in
    classification prediction image

    Args:
    y_pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for
    image #ind in the batch where annot_type = 0 or 1: 0 - contains_dot (from classification head),
    1 - offset matrices (from regression head)

    ind: the index of the image in the batch for which to convert the annotations

    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered
    containing a spot center,and their corresponding regression values will be used to create a
    final spot position prediction which will be added to the output spot center coordinates list.

    min_distance: the minimum distance between detected spots in pixels

    Returns:
    A list of spot center coordinates of the format [[y0, x0], [y1, x1],...]
    """
    dot_centers = []
    for ind in range(np.shape(y_pred)[1]):
        dot_pixel_inds = peak_local_max(
            y_pred[1][ind, ..., 1], min_distance=min_distance, threshold_abs=threshold)

        delta_y = y_pred[0][ind, ..., 0]
        delta_x = y_pred[0][ind, ..., 1]

        dot_centers.append(np.array(
            [[y_ind + delta_y[y_ind, x_ind],
              x_ind + delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds]))

    return np.array(dot_centers)


def y_annotations_to_point_list_cc(y_pred, threshold=0.8):
    # make final decision to be: average regression over each connected component of above
    # detection threshold pixels

    dot_centers = []
    for ind in range(np.shape(y_pred)[1]):

        delta_y = y_pred[0][ind, ..., 0]
        delta_x = y_pred[0][ind, ..., 1]

        blobs = y_pred[1][ind, ..., 1] > threshold
        label_image = measure.label(blobs, background=0)
        rp = measure.regionprops(label_image)

        dot_centers_temp = []
        for region in rp:
            region_pixel_inds = region.coords
            reg_pred = [[y_ind + delta_y[y_ind, x_ind], x_ind + delta_x[y_ind, x_ind]]
                        for y_ind, x_ind in region_pixel_inds]
            dot_centers_temp.append(np.mean(reg_pred, axis=0))

        dot_centers.append(dot_centers_temp)
    return np.array(dot_centers)
