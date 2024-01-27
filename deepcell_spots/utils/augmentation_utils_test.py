# Copyright 2019-2024 The Van Valen Lab at the California Institute of
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

"""Tests for utils"""

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.utils.augmentation_utils import (affine_transform_points,
                                                     generate_transformation_matrix,
                                                     subpixel_distance_transform)


class TestUtils(test.TestCase):
    def test_subpixel_distance_transform(self):
        # test shape of output for square image
        point_list = np.random.random((10, 2))
        image_shape = (30, 30)
        delta_y, delta_x, nearest_point = subpixel_distance_transform(
            point_list, image_shape)

        self.assertEqual(np.shape(delta_y), image_shape)
        self.assertEqual(np.shape(delta_x), image_shape)
        self.assertEqual(np.shape(nearest_point), image_shape)

        # test shape of output for rectangular image
        point_list = np.random.random((10, 2))
        image_shape = (50, 30)
        delta_y, delta_x, nearest_point = subpixel_distance_transform(
            point_list, image_shape)

        self.assertEqual(np.shape(delta_y), image_shape)
        self.assertEqual(np.shape(delta_x), image_shape)
        self.assertEqual(np.shape(nearest_point), image_shape)

        # test image with no points
        point_list = np.random.random((0, 2))
        image_shape = (30, 30)
        delta_y, delta_x, nearest_point = subpixel_distance_transform(
            point_list, image_shape)

        self.assertEqual(np.shape(delta_y), image_shape)
        self.assertEqual(np.shape(delta_x), image_shape)
        self.assertEqual(np.shape(nearest_point), image_shape)

        self.assertAllEqual(delta_y, np.full(image_shape, image_shape[1]).astype(float))
        self.assertAllEqual(delta_x, np.full(image_shape, image_shape[0]).astype(float))
        self.assertAllEqual(nearest_point, np.full(image_shape, np.nan))

    def test_generate_transformation_matrix(self):
        transform_parameters = {
            'theta': 0,
            'tx': 0,
            'ty': 0,
            'shear': 0,
            'zx': 1,
            'zy': 1
        }

        image_shape = (10, 10)
        img_row_axis = 0
        img_col_axis = 1

        final_affine_matrix, final_offset = generate_transformation_matrix(
            transform_parameters,
            image_shape,
            img_row_axis,
            img_col_axis
        )

        # null transformation matrix
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])

        self.assertEqual(np.shape(final_affine_matrix), (2, 2))
        self.assertEqual(len(final_offset), 2)
        self.assertAllEqual(final_affine_matrix, transform_matrix[:2, :2])
        self.assertAllEqual(final_offset, transform_matrix[:2, 2])

        transform_parameters = {
            'theta': 5,
            'tx': 1,
            'ty': 1,
            'shear': 5,
            'zx': 5,
            'zy': 5
        }

        image_shape = (10, 10)
        img_row_axis = 0
        img_col_axis = 1

        final_affine_matrix, final_offset = generate_transformation_matrix(
            transform_parameters,
            image_shape,
            img_row_axis,
            img_col_axis
        )

        # null transformation matrix
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])

        self.assertEqual(np.shape(final_affine_matrix), (2, 2))
        self.assertEqual(len(final_offset), 2)

    def test_affine_transform_points(self):
        num_points = 10
        points = np.random.randint(low=1, high=9, size=(num_points, 2))

        transform_parameters = {
            'theta': 0,
            'tx': 0,
            'ty': 0,
            'shear': 0,
            'zx': 1,
            'zy': 1
        }

        image_shape = (10, 10)

        # Fill with nearest
        transformed_points_in_image = affine_transform_points(points,
                                                              transform_parameters,
                                                              image_shape,
                                                              fill_mode='nearest')

        self.assertEqual(np.shape(transformed_points_in_image), (num_points, 2))
        self.assertAllEqual(points, transformed_points_in_image)

        # Fill with reflect
        transformed_points_in_image = affine_transform_points(points,
                                                              transform_parameters,
                                                              image_shape,
                                                              fill_mode='reflect')

        self.assertEqual(np.shape(transformed_points_in_image), (num_points, 2))
        self.assertAllEqual(points, transformed_points_in_image)

        # Fill with wrap
        transformed_points_in_image = affine_transform_points(points,
                                                              transform_parameters,
                                                              image_shape,
                                                              fill_mode='wrap')

        self.assertEqual(np.shape(transformed_points_in_image), (num_points, 2))
        self.assertAllEqual(points, transformed_points_in_image)


if __name__ == '__main__':
    test.main()
