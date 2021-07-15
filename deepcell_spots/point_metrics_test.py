import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.spatial  # cKDTree - neighbor finder (cython)
from sklearn.metrics import *

from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from point_metrics import *
from postprocessing_utils import *


class TestPointMetrics(test.TestCase):
    def test_sum_min_distance(self):
        # Example with no spots
        coords1 = np.array([])
        coords2 = np.array([])
        d_md = sum_of_min_distance(coords1, coords2)
        self.assertEqual(d_md, np.inf)

        # Example with exact matching lists
        coords1 = np.array([[1, 1], [2, 2], [3, 3]])
        coords2 = np.array([[1, 1], [2, 2], [3, 3]])
        d_md = sum_of_min_distance(coords1, coords2)
        self.assertEqual(d_md, 0)

        # Example with easy to calculate error
        coords1 = np.array([[1, 1], [3, 3], [5, 5]])
        coords2 = np.array([[1.5, 1.5], [3.5, 3.5], [5.5, 5.5]])
        d_md = sum_of_min_distance(coords1, coords2)
        self.assertEqual(d_md, 3*0.5*np.sqrt(2))

        # Example with normalized error
        d_md = sum_of_min_distance(coords1, coords2, normalized=True)
        self.assertEqual(d_md, 0.5*np.sqrt(2))

    def test_match_points_min_dist(self):
        # Two matching points
        pts1 = np.array([[0, 0], [2, 2]])
        pts2 = np.array([[2, 2], [0, 0]])
        row_ind, col_ind = match_points_min_dist(pts1, pts2, threshold=0.5)

        self.assertEqual(len(pts1), len(row_ind))
        self.assertEqual(len(row_ind), len(col_ind))
        self.assertEqual(row_ind.all(), np.array([0, 1]).all())
        self.assertEqual(col_ind.all(), np.array([1, 0]).all())

        # One matching point, and one mis-matched point
        pts1 = np.array([[0, 0], [2, 2]])
        pts2 = np.array([[3, 3], [0, 0]])
        row_ind, col_ind = match_points_min_dist(pts1, pts2, threshold=0.5)

        self.assertGreater(len(pts1), len(row_ind))
        self.assertEqual(len(row_ind), len(col_ind))
        self.assertEqual(row_ind.all(), np.array([0]).all())
        self.assertEqual(col_ind.all(), np.array([1]).all())

        # One matching point, and one mis-matched point -- no threshold
        pts1 = np.array([[0, 0], [2, 2]])
        pts2 = np.array([[3, 3], [0, 0]])
        row_ind, col_ind = match_points_min_dist(pts1, pts2, threshold=None)

        self.assertEqual(len(pts1), len(row_ind))
        self.assertEqual(len(row_ind), len(col_ind))
        self.assertEqual(row_ind.all(), np.array([0, 1]).all())
        self.assertEqual(col_ind.all(), np.array([1, 0]).all())

    def test_match_points_mutual_nearest_neighbor(self):
        # Two matching points
        pts1 = np.array([[0, 0], [2, 2]])
        pts2 = np.array([[2, 2], [0, 0]])
        row_ind, col_ind = match_points_mutual_nearest_neighbor(
            pts1, pts2, threshold=0.5)

        self.assertEqual(len(pts1), len(row_ind))
        self.assertEqual(len(row_ind), len(col_ind))
        self.assertEqual(row_ind.all(), np.array([0, 1]).all())
        self.assertEqual(col_ind.all(), np.array([1, 0]).all())

        # One matching point, and one mis-matched point
        pts1 = np.array([[0, 0], [2, 2]])
        pts2 = np.array([[3, 3], [0, 0]])
        row_ind, col_ind = match_points_mutual_nearest_neighbor(
            pts1, pts2, threshold=0.5)

        self.assertGreater(len(pts1), len(row_ind))
        self.assertEqual(len(row_ind), len(col_ind))
        self.assertEqual(row_ind.all(), np.array([0]).all())
        self.assertEqual(col_ind.all(), np.array([1]).all())

        # One matching point, and one mis-matched point -- no threshold
        pts1 = np.array([[0, 0], [2, 2]])
        pts2 = np.array([[3, 3], [0, 0]])
        row_ind, col_ind = match_points_mutual_nearest_neighbor(
            pts1, pts2, threshold=None)

        self.assertEqual(len(pts1), len(row_ind))
        self.assertEqual(len(row_ind), len(col_ind))
        self.assertEqual(row_ind.all(), np.array([0, 1]).all())
        self.assertEqual(col_ind.all(), np.array([1, 0]).all())

    def test_point_precision(self):
        # Easy to calculate examples
        points_true = np.array([[1, 1]])
        points_pred = np.array([[1.5, 1.5]])
        prec = point_precision(points_true, points_pred, threshold=1)

        self.assertEqual(prec, 1)

        points_true = np.array([[1, 1]])
        points_pred = np.array([[1.5, 1.5], [3, 3]])
        prec = point_precision(points_true, points_pred, threshold=1)

        self.assertEqual(prec, 0.5)

        # Example with no true positives
        points_true = np.array([])
        points_pred = np.array([1, 1])
        prec = point_precision(points_true, points_pred, threshold=1)

        self.assertEqual(prec, 0)

    def test_point_recall(self):
        # Easy to calculate examples
        points_true = np.array([[1, 1]])
        points_pred = np.array([[1.5, 1.5]])
        recall = point_recall(points_true, points_pred, threshold=1)

        self.assertEqual(recall, 1)

        points_true = np.array([[1, 1], [3, 3]])
        points_pred = np.array([[1.5, 1.5]])
        recall = point_recall(points_true, points_pred, threshold=1)

        self.assertEqual(recall, 0.5)

        # Example with no detected points
        points_true = np.array([1, 1])
        points_pred = np.array([])
        recall = point_recall(points_true, points_pred, threshold=1)

        self.assertEqual(recall, 0)

    def test_point_F1_score(self):
        # Easy to calculate examples
        points_true = np.array([[1, 1]])
        points_pred = np.array([[1.5, 1.5]])
        f1 = point_F1_score(points_true, points_pred, threshold=1)

        self.assertEqual(f1, 1)

        points_true = np.array([[1, 1], [3, 3]])
        points_pred = np.array([[1.5, 1.5]])
        f1 = point_F1_score(points_true, points_pred, threshold=1)

        self.assertEqual(f1, 2/3)

        # Example with no detected points
        points_true = np.array([1, 1])
        points_pred = np.array([])
        f1 = point_F1_score(points_true, points_pred, threshold=1)

        self.assertEqual(f1, 0)

    def test_stats_points(self):
        # Example with easy to calculate stats
        points_true = np.array([[1, 1], [3, 3]])
        points_pred = np.array([[1.5, 1.5], [3.5, 3.5]])
        stats_dict = stats_points(points_true, points_pred, threshold=1)

        self.assertEqual(stats_dict['precision'], 1)
        self.assertEqual(stats_dict['recall'], 1)
        self.assertEqual(stats_dict['F1'], 1)
        self.assertEqual(stats_dict['JAC'], 1)
        self.assertEqual(stats_dict['RMSE'], 0.5)
        self.assertEqual(stats_dict['d_md'], 0.5*np.sqrt(2))

        # Example with no predicted points
        points_true = np.array([[1, 1], [3, 3]])
        points_pred = np.array([])
        stats_dict = stats_points(points_true, points_pred, threshold=1)

        self.assertEqual(stats_dict['precision'], 0)
        self.assertEqual(stats_dict['recall'], 0)
        self.assertEqual(stats_dict['F1'], 0)
        self.assertEqual(stats_dict['JAC'], 0)
        self.assertEqual(stats_dict['RMSE'], None)
        self.assertEqual(stats_dict['d_md'], None)

        # Example with one false positive
        points_true = np.array([[1, 1], [3, 3]])
        points_pred = np.array([[5, 5]])
        stats_dict = stats_points(points_true, points_pred, threshold=1)

        self.assertEqual(stats_dict['precision'], 0)
        self.assertEqual(stats_dict['recall'], 0)
        self.assertEqual(stats_dict['F1'], 0)
        self.assertEqual(stats_dict['JAC'], 0)
        self.assertEqual(stats_dict['RMSE'], None)
        self.assertEqual(stats_dict['d_md'], None)


if __name__ == '__main__':
    test.main()
