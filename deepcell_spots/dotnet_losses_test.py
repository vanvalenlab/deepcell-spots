"""Tests for loss functions for DeepCell spots"""

import numpy as np
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
