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

"""Tests for analysis of multiplex FISH images"""

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.multiplex import (
    multiplex_match_spots_to_cells,
    extract_spots_prob_from_coords_maxpool,
)


class TestImageAlignment(test.TestCase):
    def test_multiplex_match_spots_to_cells(self):
        coords_dict = {0: [[[0, 0], [1, 1]]]}
        cytoplasm_pred = np.zeros((1, 10, 10, 1))

        spots_dict = multiplex_match_spots_to_cells(coords_dict, cytoplasm_pred)

        print(spots_dict)
        self.assertEqual(list(spots_dict.keys()), [0])
        self.assertEqual(spots_dict[0], {0.0: [[0, 0], [1, 1]]})

    def test_extract_spots_prob_from_coords_maxpool(self):
        image = np.random.rand(10, 100, 100, 20)
        spots_locations = np.random.randint(0, 100, (20, 2))

        with self.assertRaises(ValueError):
            extract_spots_prob_from_coords_maxpool(image, spots_locations, extra_pixel_num=-1)

        with self.assertRaises(ValueError):
            extract_spots_prob_from_coords_maxpool(image, spots_locations, extra_pixel_num=0.5)


if __name__ == '__main__':
    test.main()
