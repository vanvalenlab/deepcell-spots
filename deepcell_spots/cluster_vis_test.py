import numpy as np
from scipy.spatial import distance
from itertools import combinations

from tensorflow.python.platform import test
from cluster_vis import jitter


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
