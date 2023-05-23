# Copyright 2019-2023 The Van Valen Lab at the California Institute of
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

"""Tests for postprocessing_utils"""

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.utils.postprocessing_utils import (
    y_annotations_to_point_list,
    y_annotations_to_point_list_cc,
    y_annotations_to_point_list_max,
    y_annotations_to_point_list_restrictive,
    max_cp_array_to_point_list_max,
)


class TestPostProcUtils(test.TestCase):
    def test_y_annotations_to_points_list(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output does not need to be below some threshold value (0.5), regression
        # adds to coords
        num_images = 10
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][:, 1, 1, 1] = np.ones(num_images)
        y_pred[keys[0]] = np.ones((num_images, image_dim, image_dim, 2))
        threshold = 0.9
        coords = y_annotations_to_point_list(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 2)

    def test_y_annotations_to_points_list_restrictive(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list_restrictive(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output needs to be below some threshold value (0.5)
        num_images = 1
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][:, 1, 1, 1] = np.ones(num_images)
        y_pred[keys[0]] = np.ones((num_images, image_dim, image_dim, 2))
        threshold = 0.9

        coords = y_annotations_to_point_list_restrictive(y_pred, threshold)

        for i in range(len(coords)):
            self.assertEqual(len(coords[i]), 0)

        # Regression output needs to be below some threshold value (0.5), regression adds to coords
        num_images = 10
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][:, 1, 1, 1] = np.ones(num_images)
        y_pred[keys[0]] = np.ones((num_images, image_dim, image_dim, 2)) * 0.4
        threshold = 0.9
        coords = y_annotations_to_point_list(y_pred, threshold)

        print(coords[0][0])
        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1.4)

    def test_y_annotations_to_point_list_max(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list_max(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output does not need to be below some threshold value (0.5),
        # regression adds to coords
        num_images = 10
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][:, 1, 1, 1] = np.ones(num_images)
        y_pred[keys[0]] = np.ones((num_images, image_dim, image_dim, 2))
        threshold = 0.9
        coords = y_annotations_to_point_list_max(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 2)

    def test_y_annotations_to_point_list_cc(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list_cc(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output does not need to be below some threshold value (0.5),
        # regression adds to coords
        num_images = 10
        image_dim = 10
        keys = ["offset_regression", "classification"]
        y_pred = {key: np.zeros((num_images, image_dim, image_dim, 2)) for key in keys}
        y_pred[keys[1]][:, 1, 1, 1] = np.ones(num_images)
        y_pred[keys[0]] = np.ones((num_images, image_dim, image_dim, 2))
        threshold = 0.9
        coords = y_annotations_to_point_list_cc(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 2)

    def test_max_cp_array_to_point_list_max(self):
        num_images = 2
        image_dim = 10
        max_cp_array = np.zeros((num_images, image_dim, image_dim))
        max_cp_array[0, 5, 5] = 1
        max_cp_array[1, 7, 7] = 0.95
        max_cp_array[1, 3, 3] = 0.91
        threshold = 0.9
        min_distance = 2
        dot_centers = max_cp_array_to_point_list_max(max_cp_array, threshold, min_distance)

        self.assertEqual(dot_centers[0][0][0], 5)
        self.assertEqual(dot_centers[0][0][1], 5)

        self.assertEqual(dot_centers[1][0][0], 7)
        self.assertEqual(dot_centers[1][0][1], 7)

        self.assertEqual(dot_centers[1][1][0], 3)
        self.assertEqual(dot_centers[1][1][1], 3)


if __name__ == "__main__":
    test.main()
