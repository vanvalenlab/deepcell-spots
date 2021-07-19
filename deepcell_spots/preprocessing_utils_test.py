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

"""Tests for preprocessing_utils"""

import random
import numpy as np
from tensorflow.python.platform import test

from preprocessing_utils import *


class TestPreProcUtils(test.TestCase):
    def test_mean_std_normalize(self):
        image_dims = 128
        image = np.random.random((2, image_dims, image_dims, 1))
        norm_image = mean_std_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)

        # test convert to int
        image_dims = 128
        image = np.ones((2, image_dims, image_dims, 1)).astype(int)
        norm_image = mean_std_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)

    def test_min_max_normalize(self):
        image_dims = 128
        image = np.random.random((2, image_dims, image_dims, 1))
        norm_image = min_max_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)

        # test convert to int
        image_dims = 128
        image = np.ones((2, image_dims, image_dims, 1)).astype(int)
        norm_image = min_max_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)


if __name__ == '__main__':
    test.main()
