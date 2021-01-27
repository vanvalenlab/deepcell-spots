import numpy as np
import networkx as nx 
from scipy.spatial import distance
from itertools import combinations

from tensorflow.python.platform import test
from cluster_vis import *

class TestDataUtils(test.TestCase):
    def test_define_edges(self):
        coords = np.ones((2,2))
        threshold = 0.5
        A = define_edges(coords,threshold)

        self.assertEqual(np.shape(A),(len(coords),len(coords)))
        expected_output = np.zeros((2,2))
        expected_output[0,1] += 1
        expected_output[1,0] += 1
        for i in range(len(coords)):
            for ii in range(len(coords)):
                self.assertEqual(A[i][ii],expected_output[i][ii])

        threshold = 0
        A = define_edges(coords,threshold)
        self.assertEqual(A.all(), np.zeros((2,2)).all())

    def test_jitter(self):
        coords = np.zeros((10,2))
        size = 5
        noisy_coords = jitter(coords,size)
        self.assertEqual(np.shape(coords),np.shape(noisy_coords))
        self.assertNotEqual(coords.all(),noisy_coords.all())

    


test.main()