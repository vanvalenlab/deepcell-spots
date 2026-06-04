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
