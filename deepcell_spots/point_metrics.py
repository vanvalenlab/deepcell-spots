"""Custom metrics for comparison of sets of points
A set of points is an unordered collection of points defined here by a list of the point
coordinates. The metrics defined here quantify the similarity between two sets of points,
taking into account their spatial structure. Specifically, the distance between points is
taken into account (as opposed to the Jaccard distance for example)
"""
import numpy as np
import scipy.spatial  # cKDTree - neighbor finder (cython)


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


def point_precision(points_true, points_pred, threshold):
    """ Calculates the precision, tp/(tp + fp), of point detection using the following definitions:
    true positive (tp) = a predicted dot p with distance to nearest true dot t <= thresh AND
    p and t are mutual nearest neighbors of each other from the other set (meaning p is the closest
    predicted dot to t, and t is the closest true point to p)
    otherwise, the predicted dot is a false positive (fp)
    The precision is equal to (the number of true positives) / (total number of predicted points)
    
    Args:
        points_true ((N,d) numpy.array): ground truth points for a single image
        points_pred ((N,d) numpy.array): predicted points for a single image
            where N is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp

    Returns:
        float: the precision as defined above (a number between 0 and 1)
    """
    if len(points_true) == 0 or len(points_pred) == 0:
        return 0

    # calculate the distances between true points and their nearest predicted points
    # and the distances between predicted points and their nearest true points
    tree1 = scipy.spatial.cKDTree(points_true, leafsize=2)
    dist_to_nearest_gt, nearest_gt_ind = tree1.query(points_pred)
    # dist_to_nearest_gt[i] is the distance between true positive point i and the predicted point closest to it
    # nearest_gt_ind[i] is equal to j if points_true[j] is the nearest to points_pred[i] from all of points_true
    tree2 = scipy.spatial.cKDTree(points_pred, leafsize=2)
    dist_to_nearest_pred, nearest_pred_ind = tree2.query(points_true)
    # dist_to_nearest_pred[i] is the distance between predicted point i and the true point nearest to it
    # nearest_pred_ind[i] is equal to j if points_pred[j] is the nearest to points_true[i] from all of points_pred
     
    # calculate the number of true positives
    pred_has_mutual_nn = nearest_pred_ind[nearest_gt_ind] == list(range(len(nearest_gt_ind)))
    pred_close_enough_to_nn = dist_to_nearest_gt <= threshold
    # number of true positives = number of mutual nearest neighbor pairs that are closer than threshold
    tp = sum(np.bitwise_and(pred_has_mutual_nn, pred_close_enough_to_nn))
    
    precision = tp / len(points_pred)
    return precision


def point_recall(points_true, points_pred, threshold):
    """Calculates the recall, tp/(tp + fn), of point detection using the following definitions:
    true positive (tp) = a true dot t with distance to nearest predicted dot p <= threshold
    AND p and t are mutual nearest neighbors of each other from the other set (meaning p is the
    closest predicted dot to t, and t is the closest true point to p)
    otherwise, the predicted dot is a false negative (fn)
    the recall is equal to (the number of true positives) / (total number of true points)

    Args:
        points_true ((N,d) numpy.array): ground truth points for a single image
        points_pred ((N,d) numpy.array): predicted points for a single image
            where N is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp

    Returns:
        float: the recall as defined above (a number between 0 and 1)
    """
    if len(points_true) == 0 or len(points_pred) == 0:
        return 0
    
    # calculate the distances between true points and their nearest predicted points
    # and the distances between predicted points and their nearest true points
    tree1 = scipy.spatial.cKDTree(points_true, leafsize=2)
    dist_to_nearest_gt, nearest_gt_ind = tree1.query(points_pred)
    # dist_to_nearest_gt[i] is the distance between true positive point i and the predicted point closest to it
    # nearest_gt_ind[i] is equal to j if points_true[j] is the nearest to points_pred[i] from all of points_true
    tree2 = scipy.spatial.cKDTree(points_pred, leafsize=2)
    dist_to_nearest_pred, nearest_pred_ind = tree2.query(points_true)
    # dist_to_nearest_pred[i] is the distance between predicted point i and the true point nearest to it
    # nearest_pred_ind[i] is equal to j if points_pred[j] is the nearest to points_true[i] from all of points_pred
  
    # calculate the number of true positives
    gt_has_mutual_nn = nearest_gt_ind[nearest_pred_ind] == list(range(len(nearest_pred_ind)))
    gt_close_enough_to_nn = dist_to_nearest_pred <= threshold
    tp = sum(np.bitwise_and(gt_has_mutual_nn, gt_close_enough_to_nn)) # number of true positives = number of mutual nearest neighbor pairs that are closer than thresh
    
    recall = tp / len(points_true)
    return recall


def point_F1_score(points_true, points_pred, threshold):
    """Calculates the F1 score of dot detection using the following definitions:
    F1 score = 2*p*r / (p+r)
    where
    p = precision = (the number of true positives) / (total number of predicted points)
    r = recall = (the number of true positives) / (total number of true points)
    and 
    true positive (tp) = a true dot t with distance to nearest predicted dot p <= threshold AND
    p and t are mutual nearest neighbors of each other from the other set (meaning p is the
    closest predicted dot to t, and t is the closest true point to p)
    otherwise, the predicted dot is a false negative (fn)
    the recall is equal to (the number of true positives) / (total number of true points)
    
    Args:
        points_true ((N,d) numpy.array): ground truth points for a single image
        points_pred ((N,d) numpy.array): predicted points for a single image
            where N is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp

    Returns:
        float: the F1 score as defined above (a number between 0 and 1)
    """
    p = point_precision(points_true, points_pred, threshold)
    r = point_recall(points_true, points_pred, threshold)
    if p == 0 or r == 0:
        return 0
    F1 = 2*p*r / (p+r)
    return F1


def stats_points(points_true, points_pred, threshold):
    """Calculates point-based statistics
    (sum_of_min_distance, precision, recall, F1)

    Args:
        points_true ((N,d) numpy.array): ground truth points for a single image
        points_pred ((N,d) numpy.array): predicted points for a single image
            where N is the number of points and d is the dimension
        threshold (float): a distance threshold used in the definition of tp and fp

    Returns:
        dictionary: containing the calculated statistics
    """

    d_md = sum_of_min_distance(points_true, points_pred, normalized=True)
    p = point_precision(points_true, points_pred, threshold)
    r = point_recall(points_true, points_pred, threshold)

    if p == 0 or r == 0:
        F1 = 0
    else:
        F1 = 2*p*r / (p+r)

    return {
        'd_md': d_md,
        'precision': p,
        'recall': r,
        'Fmeasure': F1
    }
