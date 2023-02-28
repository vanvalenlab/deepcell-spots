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

"""Tests for decoding functions"""

import torch
import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.decoding_functions import (reshape_torch_array, decoding_function,
                                               normalize_spot_values, kronecker_product,
                                               instantiate_rb_params, instantiate_gaussian_params)


class TestDecodingFunc(test.TestCase):
    def test_reshape_torch_array(self):
        # number of barcodes = 10, rounds = 2, channels = 3
        k = 10
        r = 2
        c = 3
        torch_array = torch.zeros(k, r, c)
        reshaped_array = reshape_torch_array(torch_array)
        self.assertEqual(reshaped_array.shape, (k, r*c))


    def test_normalize_spot_values(self):
        # number of spots = 100, rounds = 2, channels = 3
        n = 100
        r = 2
        c = 3
        data = torch.zeros(n, r*c)
        norm_data = normalize_spot_values(data)
        self.assertEqual(data.shape, norm_data.shape)

    
    def test_kronecker_product(self):
        dim = 3
        a = torch.zeros(dim, dim)
        b = torch.zeros(dim, dim)
        product_array = kronecker_product(a, b)
        self.assertEqual(product_array.shape, (dim**2, dim**2))

        a = torch.tensor([[1,2], [3,4]])
        b = torch.tensor([[0,5], [6,7]])
        product_array = kronecker_product(a, b)
        expected_product = torch.tensor([[0,5,0,10],
                                         [6,7,12,14],
                                         [0,15,0,20],
                                         [18,21,24,28]])
        self.assertAllEqual(product_array, expected_product)

    
    def test_instantiate_rb_params(self):
        # number of barcodes = 2, rounds = 2, channels = 3
        r = 2
        c = 3
        codes = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])

        # params mode: 1
        params_mode = "1"
        scaled_sigma, aug_temperature = instantiate_rb_params(r, c, codes, params_mode)
        # self.assertEqual(scaled_sigma.shape, (1))
        # self.assertEqual(aug_temperature.shape, (1))

    def test_decoding_function(self):
        # number of samples = 100, rounds = 2, channels = 3, barcodes = 2
        n = 100
        r = 2
        c = 3
        k = 2
        spots = np.random.rand(n, r, c)
        codes = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]).reshape(k, r, c)
        results = decoding_function(spots, codes, num_iter=20, batch_size=n, params_mode='2')
        self.assertIsInstance(results, dict)
        self.assertEqual(results["class_probs"].shape, (n, k))
        self.assertIsInstance(results["params"], dict)


if __name__ == "__main__":
    test.main()
