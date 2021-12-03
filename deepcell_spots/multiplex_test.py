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

"""Tests for analysis of multiplex FISH images"""

from tensorflow.python.platform import test
# from deepcell_spots.multiplex import *
from multiplex import *


class TestImageAlignment(test.TestCase):
    def test_multiplex_match_spots_to_cells(self):
        coords_dict = {0: [[0, 0], [1, 1]]}
        cytoplasm_pred = np.zeros((10, 10))

        spots_dict = multiplex_match_spots_to_cells(coords_dict, cytoplasm_pred)

        self.assertEqual(list(spots_dict.keys()), [0])
        self.assertEqual(spots_dict[0], [[0, 0], [1, 1]])

    # def test_cluster_points(self):



if __name__ == '__main__':
    test.main()
