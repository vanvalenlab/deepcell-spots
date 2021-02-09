import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.spatial  # cKDTree - neighbor finder (cython)

from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from point_metrics import *

class TestDataUtils(test.TestCase):
    def test_match_points_min_dist(self):
        # Two matching points
        pts1 = np.array([[0,0],[2,2]])
        pts2 = np.array([[2,2],[0,0]])
        row_ind,col_ind = match_points_min_dist(pts1,pts2,threshold=0.5)

        self.assertEqual(len(pts1),len(row_ind))
        self.assertEqual(len(row_ind),len(col_ind))
        self.assertEqual(row_ind.all(),np.array([0,1]).all())
        self.assertEqual(col_ind.all(),np.array([1,0]).all())

        # One matching point, and one mis-matched point
        pts1 = np.array([[0,0],[2,2]])
        pts2 = np.array([[3,3],[0,0]])
        row_ind,col_ind = match_points_min_dist(pts1,pts2,threshold=0.5)

        self.assertGreater(len(pts1),len(row_ind))
        self.assertEqual(len(row_ind),len(col_ind))
        self.assertEqual(row_ind.all(),np.array([0]).all())
        self.assertEqual(col_ind.all(),np.array([1]).all())

    def test_match_points_mutual_nearest_neighbor(self):
        # Two matching points
        pts1 = np.array([[0,0],[2,2]])
        pts2 = np.array([[2,2],[0,0]])
        row_ind,col_ind = match_points_mutual_nearest_neighbor(pts1,pts2,threshold=0.5)

        self.assertEqual(len(pts1),len(row_ind))
        self.assertEqual(len(row_ind),len(col_ind))
        self.assertEqual(row_ind.all(),np.array([0,1]).all())
        self.assertEqual(col_ind.all(),np.array([1,0]).all())

        # One matching point, and one mis-matched point
        pts1 = np.array([[0,0],[2,2]])
        pts2 = np.array([[3,3],[0,0]])
        row_ind,col_ind = match_points_mutual_nearest_neighbor(pts1,pts2,threshold=0.5)

        self.assertGreater(len(pts1),len(row_ind))
        self.assertEqual(len(row_ind),len(col_ind))
        self.assertEqual(row_ind.all(),np.array([0]).all())
        self.assertEqual(col_ind.all(),np.array([1]).all())

test.main()