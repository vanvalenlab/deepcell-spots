"""Custom metrics for comparison of sets of points
A set of points is an unordered collection of points defined here by a list of the point
coordinates. The metrics defined here quantify the similarity between two sets of points,
taking into account their spatial structure. Specifically, the distance between points is
taken into account (as opposed to the Jaccard distance for example)
"""

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.spatial  # cKDTree - neighbor finder (cython)
from sklearn.metrics import *
# from deepcell_spots.postprocessing_utils import *
from postprocessing_utils import *


def sum_of_min_distance(pts1, pts2, normalized=False):
    """Calculates the sum of minimal distance measure between two sets of d-dimensional points
    as suggested by Eiter and Mannila in:
    https://link.springer.com/article/10.1007/s002360050075
    Args:
       pts1 ((N1,d) numpy.array): set of N1 points in d dimensions
       pts2 ((N2,d) numpy.array): set of N2 points in d dimensions
           each row of pts1 and pts2 should be the coordinates of a single d-dimensional point
       normalized (bool): if true, each sum will be normalized by the number of elements in it,
           resulting in an intensive distance measure which doesn't scale like the number of points
    Returns:
        float: the sum of minimal distance between point sets X and Y, defined as:
        d(X,Y) = 1/2 * (sum over x in X of min on y in Y of d(x,y)
        + sum over y in Y of min on x in X of d(x,y))
        = 1/2( sum over x in X of d(x,Y) + sum over y in Y of d(X,y))
        where d(x,y) is the Euclidean distance
        Note that this isn't a metric in the mathematical sense (it doesn't satisfy the triangle
        inequality)
    """

    if len(pts1) == 0 or len(pts2) == 0:
        return np.inf

    # for each point in each of the sets, find its nearest neighbor from the other set
    tree1 = scipy.spatial.cKDTree(pts1, leafsize=2)
    dist21, _ = tree1.query(pts2)
    tree2 = scipy.spatial.cKDTree(pts2, leafsize=2)
    dist12, _ = tree2.query(pts1)

    if normalized:
        d_md = 0.5 * (np.mean(dist21) + np.mean(dist12))
    else:
        d_md = 0.5 * (np.sum(dist21) + np.sum(dist12))

    return d_md


def match_points_min_dist(pts1, pts2, threshold=None):
    '''Find a pairing between two sets of points that minimizes the sum of Euclidean distances
    between matched points from each set.
    Args:
        pts1 ((N1,d) numpy.array): a set of N1 points in d dimensions
        pts2 ((N2,d) numpy.array): a set of N2 points in d dimensions
            where N1/N2 is the number of points and d is the dimension
        threshold (float): a distance threshold for matching two points. Points that are more than
        the threshold distance apart, cannot be matched
    Returns:
        row_ind, col_ind (arrays):
        An array of row indices and one of corresponding column indices giving the optimal
        assignment, as described in:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    '''

    d = cdist(pts1, pts2, 'euclidean')

    if threshold is None:
        row_ind, col_ind = linear_sum_assignment(d)

    else:
        N1 = len(pts1)
        N2 = len(pts2)

        # build a (N1+N2) x (N1+N2) cost matrix for matching points closer than the threshold
        # each point will have an "imaginary point" at the threshold distance away, which is
        # infinitely away from all other real points. Thus if there is no closer real point,
        # the point will remain unmatchd (to a real pt).
        cost = np.zeros((N1 + N2,) * 2, dtype=float)
        cost[:N1, :N2] = d

        M2 = np.full((N2, N2), np.inf)
        np.fill_diagonal(M2, threshold)
        cost[N1:, :N2] = M2

        M1 = np.full((N1, N1), np.inf)
        np.fill_diagonal(M1, threshold)
        cost[:N1, N2:] = M1

        cost[N1:, N2:] = d.T

        row_ind, col_ind = linear_sum_assignment(cost)

        real_matches_inds = (row_ind < N1) & (col_ind < N2)
        row_ind, col_ind = row_ind[real_matches_inds], col_ind[real_matches_inds]

    return row_ind, col_ind


def match_points_mutual_nearest_neighbor(pts1, pts2, threshold=None):
    '''Find a pairing between two sets of points that ensures that each pair of points are mutual
    nearest neighbors.
    Args:
        pts1 ((N1,d) numpy.array): a set of N1 points in d dimensions
        pts2 ((N2,d) numpy.array): a set of N2 points in d dimensions
            where N1/N2 is the number of points and d is the dimension
        threshold (float): a distance threshold for matching two points. Points that are more than
        the threshold distance apart, cannot be matched
    Returns:
        row_ind, col_ind (arrays):
        An array of row indices and one of corresponding column indices giving the optimal
        assignment, as described in:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    '''
    # calculate the distances between true points and their nearest predicted points
    # and the distances between predicted points and their nearest true points
    tree1 = scipy.spatial.cKDTree(pts1, leafsize=2)
    dist_to_nearest1, nearest_ind1 = tree1.query(pts2)
    # dist_to_nearest1[i] is the distance between pts1 point i and the pts2 point closest to it
    # nearest_ind1[i] is equal to j if pts1[j] is the nearest to pts2[i] from all of pts1
    tree2 = scipy.spatial.cKDTree(pts2, leafsize=2)
    dist_to_nearest2, nearest_ind2 = tree2.query(pts1)
    # dist_to_nearest2[i] is the distance between pts2 point i and the pts1 point nearest to it
    # nearest_ind2[i] is equal to j if pts2[j] is the nearest to pts1[i] from all of pts2

    # calculate the number of true positives
    pt_has_mutual_nn2 = nearest_ind2[nearest_ind1] == list(
        range(len(nearest_ind1)))
    pt_has_mutual_nn1 = nearest_ind1[nearest_ind2] == list(
        range(len(nearest_ind2)))
    if threshold is None:
        row_ind = np.where(pt_has_mutual_nn1)[0]
        col_ind = nearest_ind2[pt_has_mutual_nn1]

    else:
        pt_close_enough_to_nn1 = dist_to_nearest2 <= threshold
        matched_pts1 = pt_has_mutual_nn1 & pt_close_enough_to_nn1
        col_ind = nearest_ind2[matched_pts1]
        row_ind = np.where(matched_pts1)[0]

    return row_ind, col_ind


def point_precision(points_true,
                    points_pred,
                    threshold,
                    match_points_function=match_points_mutual_nearest_neighbor):
    """ Calculates the precision, tp/(tp + fp), of point detection using the following definitions:
    true positive (tp) = a predicted dot p with a matching true dot t,
    where the matching between predicted and true points is such that the total distance between
    matched points is minimized, and points can be matched only if the distance between them is
    smaller than the threshold. Otherwise, the predicted dot is a false positive (fp).
    The precision is equal to (the number of true positives) / (total number of predicted points)
    Args:
        points_true ((N1,d) numpy.array): ground truth points for a single image
        points_pred ((N2,d) numpy.array): predicted points for a single image
            where N1/N2 is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp
        match_points_function: a function that matches points in two sets,
        and has three parameters: pts1, pts2, threshold -
        two sets of points, and a threshold distance for allowing a match
        supported matching functions are match_points_min_dist, match_points_mutual_nearest_neighbor
    Returns:
        float: the precision as defined above (a number between 0 and 1)
    """
    if len(points_true) == 0 or len(points_pred) == 0:
        return 0

    # find the minimal sum of distances matching between the points
    row_ind, col_ind = match_points_function(
        points_true, points_pred, threshold=threshold)

    # number of true positives = number of pairs matched
    tp = len(row_ind)

    precision = tp / len(points_pred)
    return precision


def point_recall(points_true,
                 points_pred,
                 threshold,
                 match_points_function=match_points_mutual_nearest_neighbor):
    """Calculates the recall, tp/(tp + fn), of point detection using the following definitions:
    true positive (tp) = a predicted dot p with a matching true dot t,
    where the matching between predicted and true points is such that the total distance between
    matched points is minimized, and points can be matched only if the distance between them is
    smaller than the threshold. Otherwise, the predicted dot is a false positive (fp).
    The recall is equal to (the number of true positives) / (total number of true points)
    Args:
        points_true ((N1,d) numpy.array): ground truth points for a single image
        points_pred ((N2,d) numpy.array): predicted points for a single image
            where N1/N2 is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp
    Returns:
        float: the recall as defined above (a number between 0 and 1)
    """
    if len(points_true) == 0 or len(points_pred) == 0:
        return 0

    # find the minimal sum of distances matching between the points
    row_ind, col_ind = match_points_function(
        points_true, points_pred, threshold=threshold)

    # number of true positives = number of pairs matched
    tp = len(row_ind)

    recall = tp / len(points_true)
    return recall


def point_F1_score(points_true,
                   points_pred,
                   threshold,
                   match_points_function=match_points_mutual_nearest_neighbor):
    """Calculates the F1 score of dot detection using the following definitions:
    F1 score = 2*p*r / (p+r)
    where
    p = precision = (the number of true positives) / (total number of predicted points)
    r = recall = (the number of true positives) / (total number of true points)
    and
    true positive (tp) = a predicted dot p with a matching true dot t,
    where the matching between predicted and true points is such that the total distance between
    matched points is minimized, and points can be matched only if the distance between them is
    smaller than the threshold. Otherwise, the predicted dot is a false positive (fp).
    Args:
        points_true ((N1,d) numpy.array): ground truth points for a single image
        points_pred ((N2,d) numpy.array): predicted points for a single image
            where N1/N2 is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp
    Returns:
        float: the F1 score as defined above (a number between 0 and 1)
    """
    p = point_precision(points_true, points_pred, threshold)
    r = point_recall(points_true, points_pred, threshold)
    if p == 0 or r == 0:
        return 0
    F1 = 2 * p * r / (p + r)
    return F1


def stats_points(points_true,
                 points_pred,
                 threshold,
                 match_points_function=match_points_mutual_nearest_neighbor):
    """Calculates point-based statistics
    (precision, recall, F1, JAC, RMSE, d_md)
    Args:
        points_true ((N1,d) numpy.array): ground truth points for a single image
        points_pred ((N2,d) numpy.array): predicted points for a single image
            where N1/N2 is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp
    Returns:
        dictionary: containing the calculated statistics
    """

    # if one of the point sets is empty, precision=recall=0
    if len(points_true) == 0 or len(points_pred) == 0:
        p = 0
        r = 0
        F1 = 0
        J = 0
        RMSE = None
        dmd = None

        return {
            'precision': p,
            'recall': r,
            'F1': F1,
            'JAC': J,
            'RMSE': RMSE,
            'd_md': dmd
        }

    # find the minimal sum of distances matching between the points
    row_ind, col_ind = match_points_function(
        points_true, points_pred, threshold=threshold)

    # number of true positives = number of pairs matched
    tp = len(row_ind)

    p = tp / len(points_pred)
    r = tp / len(points_true)

    # calculate the F1 score from the precision and the recall
    if p == 0 or r == 0:
        F1 = 0
    else:
        F1 = 2 * p * r / (p + r)

    # calculate the Jaccard index from the F1 score
    J = F1 / (2 - F1)

    # calculate the RMSE for matched pairs
    if len(row_ind) == 0:
        RMSE = None
        d_md = None
    else:
        dist_sq_sum = np.sum(
            np.sum((points_true[row_ind] - points_pred[col_ind]) ** 2, axis=1))
        RMSE = np.sqrt(dist_sq_sum / len(row_ind) / 2)

        # RMSE = np.sqrt(mean_squared_error(points_true[row_ind], points_pred[col_ind]))

        # calculate the mean sum to nearest neighbor from other set
        d_md = sum_of_min_distance(
            points_true[row_ind], points_pred[col_ind], normalized=True)

    return {
        'precision': p,
        'recall': r,
        'F1': F1,
        'JAC': J,
        'RMSE': RMSE,
        'd_md': d_md
    }


def get_mean_stats(y_test, y_pred, threshold=0.98, d_thresh=1):
    # decision_function: a postprocessing function with inputs y_pred, ind, threshold
    # this is a function that performs postprocessing on the output of the neural net for the image
    # ind in the batch,
    # and returns a decision for the list of coordinates of spot centers
    # d_thresh = 2 # distance in pixels for precision quantification
    # threshold = 0.95 # threshold for classification
    n_test = len(y_test)  # number of test images

    d_md_list = [None] * n_test
    precision_list = [None] * n_test
    recall_list = [None] * n_test
    F1_list = [None] * n_test

    y_pred = y_annotations_to_point_list_max(y_pred, threshold)
    for ind in range(n_test):  # loop over test images
        # y_pred_ind = y_annotations_to_point_list(y_pred_test, ind, threshold)
        s = stats_points(y_test[ind], y_pred[ind], threshold=d_thresh)
        d_md_list[ind] = s['d_md']
        precision_list[ind] = s['precision']
        recall_list[ind] = s['recall']
        F1_list[ind] = s['Fmeasure']

    d_md = np.mean(d_md_list)
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    F1 = np.mean(F1_list)

    return (d_md, precision, recall, F1)


def model_benchmarking(pred, coords, threshold, min_distance):
    """Calculates the precision, recall, F1 score, Jacard Index, root mean square error, and sum of
    min distances for stack of predictions"

    Args:
    pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for image
    ind in the batch where annot_type = 0 or 1: 0 - contains_dot (from classification head),
    1 - offset matrices (from regression head)
    coords: nested list of coordinate locations for ground truth spots from a single annotator
    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered
    containing a spot center, and their corresponding regression values will be used to create a
    final spot position prediction which will be added to the output spot center coordinates list.
    min_distance: the minimum distance between detected spots in pixels

    Returns:
    precision: list of values for the precision of the predicted spot numbers in each image
    recall: list of values for the recall of the predicted spot numbers in each image
    f1: list of values for the f1 score of the predicted spot numbers in each image
    jac: list of values for the jacard index of the predicted spot numbers in each image
    rmse: list of values for the root mean square error of the spot locations in each image
    dmd: list of values for the sum of min distance of the spot locations in each image

    """
    precision = []
    recall = []
    f1 = []
    jac = []
    rmse = []
    dmd = []

    points_list = y_annotations_to_point_list_max(
        pred, threshold, min_distance)
    for i in range(len(pred[0])):

        stats_dict = stats_points(
            coords[i],
            points_list[i],
            1,
            match_points_function=match_points_mutual_nearest_neighbor)

        precision.append(stats_dict['precision'])
        recall.append(stats_dict['recall'])
        f1.append(stats_dict['F1'])
        jac.append(stats_dict['JAC'])
        rmse.append(stats_dict['RMSE'])
        dmd.append(stats_dict['d_md'])

    return(precision, recall, f1, jac, rmse, dmd)
