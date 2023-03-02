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

import gc
import pyro
import torch
import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.decoding_functions import (reshape_torch_array, decoding_function,
                                               normalize_spot_values, kronecker_product,
                                               chol_sigma_from_vec, rb_e_step, gaussian_e_step,
                                               instantiate_rb_params, instantiate_gaussian_params)

torch.cuda.empty_cache()
gc.collect()

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

    def test_chol_sigma_from_vec(self):
        dim = 3
        sigma_vec = torch.zeros(np.sum(np.arange(dim)+1))
        output = chol_sigma_from_vec(sigma_vec, dim)
        self.assertAllEqual(output, torch.zeros(dim, dim))

        dim = 3
        sigma_vec = torch.ones(np.sum(np.arange(dim)+1))
        output = chol_sigma_from_vec(sigma_vec, dim)
        expected_tri_lower = torch.tensor([[1,0,0],[1,1,0],[1,1,1]])
        expected_tri_upper = torch.t(expected_tri_lower)
        expected_output = torch.mm(expected_tri_lower, expected_tri_upper)
        self.assertAllEqual(output, expected_output)

    def test_instantiate_rb_params(self):
        # number of barcodes = 2, rounds = 2, channels = 3
        r = 2
        c = 3
        codes = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        k = codes.shape[0]

        # params mode: 2
        params_mode = "2"
        scaled_sigma, aug_temperature = instantiate_rb_params(r, c, codes, params_mode)
        self.assertAllEqual(scaled_sigma.shape, torch.Size([k,r*c]))
        self.assertAllEqual(aug_temperature.shape, torch.Size([k,r*c]))
        pyro.get_param_store().clear()

        # params mode: 2*R
        params_mode = "2*R"
        scaled_sigma, aug_temperature = instantiate_rb_params(r, c, codes, params_mode)
        self.assertAllEqual(scaled_sigma.shape, torch.Size([k,r*c]))
        self.assertAllEqual(aug_temperature.shape, torch.Size([k,r*c]))
        pyro.get_param_store().clear()

        # params mode: 2*C
        params_mode = "2*C"
        scaled_sigma, aug_temperature = instantiate_rb_params(r, c, codes, params_mode)
        self.assertAllEqual(scaled_sigma.shape, torch.Size([k,r*c]))
        self.assertAllEqual(aug_temperature.shape, torch.Size([k,r*c]))
        pyro.get_param_store().clear()

        # params mode: 2*R*C
        params_mode = "2*R*C"
        scaled_sigma, aug_temperature = instantiate_rb_params(r, c, codes, params_mode)
        self.assertAllEqual(scaled_sigma.shape, torch.Size([k,r*c]))
        self.assertAllEqual(aug_temperature.shape, torch.Size([k,r*c]))
        pyro.get_param_store().clear()

    def test_instantiate_gaussian_params(self):
        # number of barcodes = 2, rounds = 2, channels = 3
        r = 2
        c = 3
        codes = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        k = codes.shape[0]
        theta, sigma = instantiate_gaussian_params(r, c, codes)
        self.assertAllEqual(sigma.shape, torch.Size([r*c, r*c]))
        self.assertAllEqual(theta.shape, torch.Size([k, r*c]))
        pyro.get_param_store().clear()

    def test_rb_e_step(self):
        # number of barcodes = 2, rounds = 2, channels = 3, spots = 100
        r = 2
        c = 3
        n = 100
        spots = np.random.rand(n, r*c)
        data = torch.tensor(spots)
        codes = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        k = codes.shape[0]
        w = torch.ones(k) / k

        # params mode: 2
        temperature = torch.ones(torch.Size([2])) * 0.5
        sigma = torch.ones(torch.Size([2])) * 0.5
        params_mode = '2'
        class_prob_norm = rb_e_step(data, codes, w, temperature, sigma, c, r, params_mode)
        self.assertAllEqual(class_prob_norm.shape, torch.Size([n, k]))
        self.assertAllGreater(class_prob_norm, 0)
        self.assertAllLess(class_prob_norm, 1)
        pyro.get_param_store().clear()

        # params mode: 2*R
        temperature = torch.ones(torch.Size([2,r])) * 0.5
        sigma = torch.ones(torch.Size([2,r])) * 0.5
        params_mode = '2*R'
        class_prob_norm = rb_e_step(data, codes, w, temperature, sigma, c, r, params_mode)
        self.assertAllEqual(class_prob_norm.shape, torch.Size([n, k]))
        self.assertAllGreater(class_prob_norm, 0)
        self.assertAllLess(class_prob_norm, 1)
        pyro.get_param_store().clear()

        # params mode: 2*C
        temperature = torch.ones(torch.Size([2,c])) * 0.5
        sigma = torch.ones(torch.Size([2,c])) * 0.5
        params_mode = '2*C'
        class_prob_norm = rb_e_step(data, codes, w, temperature, sigma, c, r, params_mode)
        self.assertAllEqual(class_prob_norm.shape, torch.Size([n, k]))
        self.assertAllGreater(class_prob_norm, 0)
        self.assertAllLess(class_prob_norm, 1)
        pyro.get_param_store().clear()

        # params mode: 2*R*C
        temperature = torch.ones(torch.Size([2, r*c])) * 0.5
        sigma = torch.ones(torch.Size([2, r*c])) * 0.5
        params_mode = '2*R*C'
        class_prob_norm = rb_e_step(data, codes, w, temperature, sigma, c, r, params_mode)
        self.assertAllEqual(class_prob_norm.shape, torch.Size([n, k]))
        self.assertAllGreater(class_prob_norm, 0)
        self.assertAllLess(class_prob_norm, 1)
        pyro.get_param_store().clear()

    def test_gaussian_e_step(self):
        # number of barcodes = 2, rounds = 2, channels = 3, spots = 100
        r = 2
        c = 3
        n = 100
        spots = np.random.rand(n, r*c)
        data = torch.tensor(spots)
        codes = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        k = codes.shape[0]
        w = torch.ones(k) / k
        theta = torch.zeros(k, r*c)
        sigma_c_v = torch.eye(c)[np.tril_indices(c, 0)]
        sigma_c = chol_sigma_from_vec(sigma_c_v, c)
        sigma_r_v = torch.eye(r*c)[np.tril_indices(r, 0)]
        sigma_r = chol_sigma_from_vec(sigma_r_v, r)
        sigma = kronecker_product(sigma_r, sigma_c)
        class_prob_norm = gaussian_e_step(data, w, theta, sigma, k)
        self.assertAllEqual(class_prob_norm.shape, torch.Size([n, k]))
        self.assertAllGreater(class_prob_norm, 0)
        self.assertAllLess(class_prob_norm, 1)
        pyro.get_param_store().clear()

    def test_decoding_function(self):
        # number of samples = 100, rounds = 2, channels = 3, barcodes = 2
        n = 100
        r = 2
        c = 3
        k = 2
        spots = np.random.rand(n, r, c)
        codes = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]).reshape(k, r, c)

        # Relaxed Bernoulli distributions
        results = decoding_function(spots, codes, num_iter=20, batch_size=n, params_mode='2*R*C')
        self.assertIsInstance(results, dict)
        self.assertAllEqual(results["class_probs"].shape, torch.Size([n, k]))
        self.assertIsInstance(results["params"], dict)
        pyro.get_param_store().clear()

        # Gaussian distributions
        results = decoding_function(spots, codes, num_iter=20, batch_size=n, params_mode='Gaussian')
        self.assertIsInstance(results, dict)
        self.assertAllEqual(results["class_probs"].shape, torch.Size([n, k]))
        self.assertIsInstance(results["params"], dict)


if __name__ == "__main__":
    test.main()
