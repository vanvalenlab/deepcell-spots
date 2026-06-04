"""Tests for SpotDetection application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.dotnet import dot_net_2D
from deepcell_spots.applications import SpotDetection


class TestSpotDetection(test.TestCase):

    def test_spot_detection_app(self):
        with self.cached_session():
            model = dot_net_2D(receptive_field=13,
                               input_shape=(128, 128, 1),
                               inputs=None,
                               n_skips=3,
                               norm_method=None,
                               padding_mode='reflect')

            app = SpotDetection(model)

            # test output shape
            shape = app.model.output_shape
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 2)  # 2 prediction heads
            self.assertEqual(len(shape[0]), 4)
            self.assertEqual(len(shape[1]), 4)

            # test threshold error
            app = SpotDetection(model=model)
            spots_image = np.random.rand(1, 128, 128, 1)
            with self.assertRaises(ValueError):
                _ = app.predict(image=spots_image, threshold=1.1)
            with self.assertRaises(ValueError):
                _ = app.predict(image=spots_image, threshold=-1.1)
