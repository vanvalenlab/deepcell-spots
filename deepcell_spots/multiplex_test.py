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

        with self.assertRaises(TypeError):
            extract_spots_prob_from_coords_maxpool(image, spots_locations, extra_pixel_num=0.5)


if __name__ == '__main__':
    test.main()
