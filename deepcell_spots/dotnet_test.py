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

"""Tests for CNN dotnet architechture"""

from absl.testing import parameterized

from tensorflow.python.framework import test_util as tf_test_util
from keras import keras_parameterized

from tensorflow.keras import backend as K

from deepcell_spots.dotnet import dot_net_2D


class FeatureNetTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters([
        {
            'testcase_name': 'reflect_padding',
            'padding_mode': 'reflect',
            'norm_method': 'std',
            'shape': (128, 128, 1),
            'receptive_field': 13,
            'data_format': 'channels_last'
        },
        {
            'testcase_name': 'zero_padding',
            'padding_mode': 'zero',
            'norm_method': 'std',
            'shape': (128, 128, 1),
            'receptive_field': 13,
            'data_format': 'channels_last'
        },
        {
            'testcase_name': 'no_norm',
            'padding_mode': 'reflect',
            'norm_method': None,
            'shape': (128, 128, 1),
            'receptive_field': 13,
            'data_format': 'channels_last'
        },
    ])
    def test_dot_net_2D(self, padding_mode, norm_method, shape,
                        receptive_field, data_format):

        inputs = None
        n_skips = 3

        with self.cached_session():
            K.set_image_data_format(data_format)
            model = dot_net_2D(
                receptive_field=receptive_field,
                input_shape=shape,
                inputs=inputs,
                n_skips=n_skips,
                norm_method=norm_method,
                padding_mode=padding_mode)
            self.assertEqual(len(model.output_shape), 2)
            self.assertEqual(len(model.output_shape[0]), 4)
            self.assertEqual(len(model.output_shape[1]), 4)
