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

"""Functions for image augmentation"""

import numpy as np
from keras_preprocessing.image.affine_transformations import \
    transform_matrix_offset_center
from scipy.ndimage.morphology import distance_transform_edt


def subpixel_distance_transform(point_list, image_shape, dy=1, dx=1):
    """For each pixel in image, return the vectorial distance to a point in
    ``point_list`` that is in the pixel nearest to it.

    Args:
        point_list: Array of size `(N,2)` of point coordinates `[y, x]`
            (y before x as in image/matrix indexing)
        image_shape: `(Ly,Lx)` specifies the shape of an image that contains
            the coordinates.
            The coordinates should be in ``dy*[-0.5, Ly-0.5] x dx*[-0.5, Lx-0.5]``
        dy: pixel width in y axis
        dx: pixel width in x axis

    Returns:
        numpy.array: `(Ly, Lx)`, nearest_point[i,j] is the index in point_list of
            a point in a point-containing pixel which is closest to pixel `[i,j]`.
            Note no uniqueness of the point or the pixel, since there could be
            several point-containing pixels with minimal distance to pixel `[i,j]`
            and there could be several points contained in the pixel `[i,j]` but
            only one is chosen `delta_x[i,j]`, `delta_y[i,j]` are elements of the
            vectorial distance between the chosen point which `nearest_point[i,j]`
            refers to, and the center of the pixel `[i,j]`,
            which is at ``x =j * dx, y = i * dy``.
        numpy.array: `(Ly, Lx)` numpy array of signed y distance between a point
            from `point_list` that is near pixel `[i,j]` and the center of the
            pixel.
        numpy.array: (Ly, Lx) numpy array of signed x distance between a point
            from `point_list` that is near pixel `[i,j]` and the center of the
            pixel.
    """
    # create an image with 0 = pixel containing point from point_list, 1 = pixel not containing
    # point from point_list
    contains_point = np.ones(image_shape)
    # index in point_list of point nearest to pixel
    nearest_point = np.full(image_shape, np.nan)

    # dictionary to be filled s.t.: pixel_to_contained_point_ind[(i,j)] = k if point_list[k]
    # is a point contained in pixel i,j of the image
    pixel_to_contained_point_ind = {}

    for ind, [y, x] in enumerate(point_list):
        nearest_pixel_y_ind = int(round(y / dy))
        nearest_pixel_x_ind = int(round(x / dx))
        contains_point[nearest_pixel_y_ind, nearest_pixel_x_ind] = 0
        pixel_to_contained_point_ind[(
            nearest_pixel_y_ind, nearest_pixel_x_ind)] = ind

    edt, inds = distance_transform_edt(
        contains_point, return_indices=True, sampling=[dy, dx])

    # signed y distance to nearest point
    delta_y = np.full(image_shape, np.inf)
    # signed x distance to nearest point
    delta_x = np.full(image_shape, np.inf)

    Ly, Lx = image_shape
    for j in range(0, Lx):
        for i in range(0, Ly):
            # inds[0][i,j] # y index of nearest pixel to [i,j] which contains a point
            # inds[1][i,j] # x index of nearest pixel to [i,j] which contains a point
            # nearest_point[i,j] # index in point_list of point contained by pixel [i,j]
            nearest_point[i, j] = pixel_to_contained_point_ind[(
                inds[0][i, j], inds[1][i, j])]
            delta_y[i, j] = dy * (point_list[int(nearest_point[i, j])][0] - i)
            delta_x[i, j] = dx * (point_list[int(nearest_point[i, j])][1] - j)

    return delta_y, delta_x, nearest_point


def generate_transformation_matrix(transform_parameters, image_shape, img_row_axis, img_col_axis):
    """Given a dictionary of affine transformation parameters (such as the one
    generated by the ``ImageDataGenerator`` function ``get_random_transform``),
    generate the transformation matrix and offset which ``apply_affine_transform``
    generates and passes to ``scipy.ndimage.interpolation.affine_transform``:

    .. code-block:: python

        ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval)

    this function performs the calculations performed by
    ``tf.keras.preprocessing.image.apply_affine_transform``
    to obtain `final_affine_matrix` and `final_offset`, and returns them.

    A point p in the output image of `affine_transform` corresponds to the point
    `pT+s` in the input image

    Args:
        transform_parameters: dictionary of affine transformation parameters such as
            the output of ``ImageDataGenerator`` method ``get_random_transform``.
            (as used in input to ``apply_transform`` called on image)
            From ``keras-preprocessing/keras_preprocessing/image/image_data_generator.py``
            method ``apply_transform`` documentation:
            Dictionary with string - parameter pairs describing the transformation.
            Currently, the following parameters
            from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip. - NOT USED HERE
                - `'flip_vertical'`: Boolean. Vertical flip. - NOT USED HERE
                - `'channel_shift_intensity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

    Returns:
        (array, array): final_affine_matrix (2*2 matrix ,denote below: T),
        final_offset (length 2 vector, denote below: s)
    """
    # get transform parameters from the input dictionary (if non given, the default value in
    # apply_affine_transform is used)
    theta = transform_parameters.get('theta', 0)
    tx = transform_parameters.get('tx', 0)
    ty = transform_parameters.get('ty', 0)
    shear = transform_parameters.get('shear', 0)
    zx = transform_parameters.get('zx', 1)
    zy = transform_parameters.get('zy', 1)

    # generate the transform matrix and offset vector
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is None:
        # if no shift, shear or zoom are done, the transformation is the identity
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
    # if transform_matrix is not None:
    h, w = image_shape[img_row_axis], image_shape[img_col_axis]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    return final_affine_matrix, final_offset


def affine_transform_points(points, transform_parameters,
                            image_shape, img_row_axis=0, img_col_axis=1, fill_mode='nearest'):
    """Perform an affine transform mapping input coordinates referring to the
    input image of the ``apply_transform`` function of the class ``ImageDataGenerator``
    To the output space of that function. Returned points are (original and
    padding points) contained in the output image.

    Args:
        transform_parameters: dictionary of affine transformation parameters
            such as the output of ``ImageDataGenerator`` method ``get_random_transform``.
        points: (N, 2) numpy array which contains points in the format `[y, x]`
            (NOTE: as in image/matrix notation, not Cartesian notation)
            points are labels for the input image
            - they should be ``-0.5 <= x <= Lx-0.5, -0.5 <= y <= Ly-0.5``
            where ``Lx = image_shape[img_col_axis]`` and ``Ly = image_shape[img_row_axis]``.
        image_shape (tuple): the shape of the image which contains the points
            (for 2D image, has length 2).
        img_row_axis: the index of the axis `(0 or 1)` to be flipped when
            `flip_vertical` is ``True``.
        img_col_axis: the index of the axis `(0 or 1)` to be flipped when
            `flip_horizontal` is ``True``.
        fill_mode: One of `("constant", "nearest", "reflect" or "wrap")`.
            Default is `'nearest'`. Points outside the boundaries of the input
            are filled according to the given mode:
                - `'constant'`: kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - `'nearest'`:  aaaaaaaa|abcd|dddddddd
                - `'reflect'`:  abcddcba|abcd|dcbaabcd
                - `'wrap'`:     abcdabcd|abcd|abcdabcd

    Returns:
        transformed_points: list of points / or numpy array of shape `(N',2)`
            which contains points in the format `[y, x]`.
            NOTE `N' != N` because points in the original image may fall outside
            of the transformed output image. Also, if fill_mode is `'reflect'` or
            `'wrap'`, point images in the padding of the input image can be
            inside the output image.
    """

    transform, offset = generate_transformation_matrix(
        transform_parameters, image_shape, img_row_axis, img_col_axis)

    # add padding point image labels to points array
    if fill_mode == 'reflect' or fill_mode == 'wrap':
        # shorten notation of image size
        Ly = image_shape[img_row_axis]
        Lx = image_shape[img_col_axis]
        # create point images in the unit cells overlapping the parallelogram domain in the input
        # space which is transformed into the output image range
        # coordinates of the corner points of the image
        corner_points = np.array([[0, 0], [0, Lx], [Ly, 0], [Ly, Lx]]) - 0.5
        # apply the output image -> input image space affine transform on the corner points, to get
        # the corner points of the parallelogram domain in input space which is mapped to the
        # output image
        input_parallelogram_corners = np.dot(
            transform, corner_points.T).T + offset
        # the input space is tiled by rectangles (unit cells) = the input image. Index them:
        # (0,0) are the y and x indices of the input image itself
        # i = ..., -2, -1, 0, 1, 2, ... are y indices for cells above and below the y position of
        #   the input image
        # j = ..., -2, -1, 0, 1, 2, ... are x indices for cells to the left and right of the input
        #   image x position
        # find the unit cells that potentially intersect with the parallelogram domain that is
        #   mapped to the output:
        y_cell_inds = input_parallelogram_corners[:, 0] // Ly
        x_cell_inds = input_parallelogram_corners[:, 1] // Lx
        y_cell_min_ind = np.min(y_cell_inds)
        y_cell_max_ind = np.max(y_cell_inds)
        x_cell_min_ind = np.min(x_cell_inds)
        x_cell_max_ind = np.max(x_cell_inds)
        # list indices of unit cells that may be transformed into the output image range
        y_cell_ind_list = range(int(y_cell_min_ind), int(y_cell_max_ind + 1))
        x_cell_ind_list = range(int(x_cell_min_ind), int(x_cell_max_ind + 1))

        # allocate empty np.array to append point images from each unit cell
        point_images = np.empty((0, 2))

        if fill_mode == 'reflect':
            for i in y_cell_ind_list:
                for j in x_cell_ind_list:
                    this_cell_points = points.copy()
                    this_cell_points[:, 0] = (
                        i * Ly + this_cell_points[:, 0]) if (i % 2 == 0) \
                        else ((i + 1) * Ly - this_cell_points[:, 0])
                    this_cell_points[:, 1] = (
                        j * Lx + this_cell_points[:, 1]) if (j % 2 == 0) \
                        else ((j + 1) * Lx - this_cell_points[:, 1])
                    point_images = np.append(
                        point_images, this_cell_points, axis=0)
        elif fill_mode == 'wrap':
            for i in y_cell_ind_list:
                for j in x_cell_ind_list:
                    this_cell_points = points.copy()
                    this_cell_points[:, 0] = this_cell_points[:, 0] + i * Ly
                    this_cell_points[:, 1] = this_cell_points[:, 1] + j * Lx
                    point_images = np.append(
                        point_images, this_cell_points, axis=0)

        transformed_points = np.dot(np.linalg.inv(
            transform), (np.array(point_images) - offset).T)

    else:  # no point images added, transform just the input points
        transformed_points = np.dot(np.linalg.inv(
            transform), (np.array(points) - offset).T)

    # apply horizontal and vertical flip (if needed)
    def flip_axis_point(x, image_shape_axis):
        """flip coordinate x around center of `(-0.5, image_shape_axis-0.5)`
        This gives the coordinate matching x in a flipped image with
        `shape[axis] == image_shape_axis`.
        """
        return image_shape_axis - 1 - x

    # transpose points back to point per row format
    transformed_points = transformed_points.T

    if transform_parameters.get('flip_horizontal', False):
        transformed_points[:, img_col_axis] = flip_axis_point(
            transformed_points[:, img_col_axis], image_shape[img_col_axis])

    if transform_parameters.get('flip_vertical', False):
        transformed_points[:, img_row_axis] = flip_axis_point(
            transformed_points[:, img_row_axis], image_shape[img_row_axis])

    # delete transformed points that are not inside the output image
    def point_in_image(points, image_shape, img_row_axis, img_col_axis):
        p_in_y = (-0.5 <= points[:, img_row_axis]) & \
            (points[:, img_row_axis] <= (image_shape[img_row_axis] - 0.5))
        p_in_x = (-0.5 <= points[:, img_col_axis]) & \
            (points[:, img_col_axis] <= (image_shape[img_col_axis] - 0.5))
        res = p_in_y & p_in_x
        return res

    transformed_points_in_image = transformed_points[point_in_image(
        transformed_points, image_shape, img_row_axis, img_col_axis), :]

    return transformed_points_in_image
