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

"""Tests for loss functions for DeepCell spots"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import test

from deepcell_spots import dotnet_losses


losses = dotnet_losses.DotNetLosses()

DOTNET_LOSSES = [
    losses.regression_loss,
    losses.classification_loss,
    losses.classification_loss_regularized
]


class KerasLossesTest(test.TestCase):

    def test_objective_shapes_4d(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.random((5, 6, 7, 8)))
            y_b = keras.backend.variable(np.random.random((5, 6, 7, 8)))

            # differs from deepcell.losses.smooth_l1 bc no summation over channels
            objective_output = dotnet_losses.smooth_l1(y_a, y_b)
            self.assertListEqual(objective_output.shape.as_list(), [5, 6, 7, 8])

            for obj in DOTNET_LOSSES:
                objective_output = obj(y_a, y_b)
                self.assertListEqual(objective_output.shape.as_list(), [])


if __name__ == '__main__':
    test.main()
