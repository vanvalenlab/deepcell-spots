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

"""Tests for expectation maximization cluster visualization"""

from itertools import combinations

import numpy as np
from scipy.spatial import distance
from tensorflow.python.platform import test

from deepcell_spots.cluster_vis import ca_to_adjacency_matrix, jitter


class TestClusterVis(test.TestCase):
    def test_jitter(self):
        coords = np.zeros((10, 2))
        size = 5
        noisy_coords = jitter(coords, size)
        self.assertEqual(np.shape(coords), np.shape(noisy_coords))
        self.assertNotEqual(coords.all(), noisy_coords.all())

    def test_ca_to_adjacency_matrix(self):
        num_clusters = 10
        num_annotators = 3
        ca_matrix = np.ones((num_clusters, num_annotators))
        A = ca_to_adjacency_matrix(ca_matrix)

        self.assertEqual(np.shape(A)[0], np.shape(A)[1], ca_matrix[0])


if __name__ == '__main__':
    test.main()
