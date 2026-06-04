"""Tests for analysis of singleplex FISH images"""

from collections import defaultdict

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.singleplex import (
    match_spots_to_cells,
    process_spot_dict,
    remove_nuc_spots_from_cyto,
    match_spots_to_cells_as_vec,
)


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
        spot_dict = remove_nuc_spots_from_cyto(labeled_im_nuc, labeled_im_cyto, coords)

        self.assertEqual(spot_dict, defaultdict(list, {1.0: [[0, 0], [1, 1]], 0: [[7, 7]]}))

    def test_match_spots_to_cells_as_vec(self):
        labeled_im = np.zeros((1, 10, 10, 1))
        coords = np.array([[0, 0], [1, 1]])

        assigned_cell = match_spots_to_cells_as_vec(labeled_im, coords)

        self.assertAllEqual(assigned_cell, [0, 0])


if __name__ == "__main__":
    test.main()
