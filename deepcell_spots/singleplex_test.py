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

"""Tests for analysis of singleplex FISH images"""

from collections import defaultdict

import numpy as np
from deepcell_spots.singleplex import (match_spots_to_cells, process_spot_dict,
                                       remove_nuc_spots_from_cyto)
from tensorflow.python.platform import test


class TestSingleplex(test.TestCase):
    def test_match_spots_to_cells(self):
        labeled_im = np.zeros((1, 10, 10, 1))
        coords = np.array([[0, 0], [1, 1]])

        spot_dict = match_spots_to_cells(labeled_im, coords)

        self.assertEqual(list(spot_dict.keys()), [0])
        self.assertAllEqual(spot_dict[0], [[0, 0], [1, 1]])

    def test_process_spot_dict(self):
        spot_dict = {0: [[0, 0], [1, 1]]}

        coords, cmap_list = process_spot_dict(spot_dict)

        self.assertAllEqual(coords, [[0, 0], [1, 1]])
        self.assertAllEqual(cmap_list, [0, 0])

    def test_remove_nuc_spots_from_cyto(self):
        labeled_im_nuc = np.concatenate((np.zeros((1, 5, 10, 1)), np.ones((1, 5, 10, 1))), axis=1)
        labeled_im_cyto = np.ones((1, 10, 10, 1))

        coords = [[0, 0], [1, 1], [7, 7]]
        spot_dict = remove_nuc_spots_from_cyto(labeled_im_nuc,
                                               labeled_im_cyto, coords)

        self.assertEqual(spot_dict, defaultdict(list, {1.0: [[0, 0], [1, 1]], 0: [[7, 7]]}))


if __name__ == '__main__':
    test.main()
