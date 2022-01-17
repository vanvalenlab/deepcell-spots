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

"""Functions for utils"""

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.utils import subpixel_distance_transform


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

        # are more tests needed?

    # def test_generate_transformation_matrix(self):
    #     # not sure what to test beside the shape


if __name__ == '__main__':
    test.main()
