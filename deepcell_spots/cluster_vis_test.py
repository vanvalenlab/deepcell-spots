import numpy as np
import networkx as nx 
from scipy.spatial import distance
from itertools import combinations

from tensorflow.python.platform import test
from cluster_vis import *

class TestClusterVis(test.TestCase):
    def test_jitter(self):
        coords = np.zeros((10,2))
        size = 5
        noisy_coords = jitter(coords,size)
        self.assertEqual(np.shape(coords),np.shape(noisy_coords))
        self.assertNotEqual(coords.all(),noisy_coords.all())

    


test.main()