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

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.decoding_functions import decoding_function


class TestDecodingFunc(test.TestCase):
    def test_decoding_function(self):
        # number of samples = 100, rounds = 2, channels = 3, barcodes = 2
        spots = np.random.rand(100, 2, 3)
        barcodes = np.array([[0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1]]).reshape(2, 2, 3)
        results = decoding_function(spots, barcodes, num_iter=20, batch_size=100)
        self.assertIsInstance(results, dict)
        self.assertEqual(results["class_probs"].shape, (100, 2))
        self.assertIsInstance(results["params"], dict)


if __name__ == "__main__":
    test.main()
