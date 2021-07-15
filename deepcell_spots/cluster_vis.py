import numpy as np
from scipy.spatial import distance
from itertools import combinations

# Functions for expectation maximization for spot detection


def jitter(coords, size):
    """ Add Gaussian noise to a list of coordinates for plotting when coordinates overlap.

    Parameters:
    ------------
    coords : matrix
        Matrix with dimensions (number of detections) x 2
    size : integer
        Standard deviation of the Gaussian noise distribution in pixels.

    Returns:
    ---------
    Coords with noise added to locations
    """
    result = []
    for item in coords:
        noise = np.random.normal(0, size)
        result.append(item+noise)

    return np.array(result)


def label_graph_ann(G, coords, exclude_last=False):
    """Labels the annotator associated with each node in the graph
    Parameters:
    ------------
    G : networkx graph
        Graph with edges indicating clusters of points assumed to be derived from the same ground
        truth detection
    coords : matrix
        2d-array of detected point locations for each classical algorithm used
    exclude_last : boolean
        Only set as True to exclude a point that has been included for the purpose of normalization

    Returns:
    ----------
    G_new : networkx graph
        Labeled graph
    """
    G_new = G.copy()
    num_spots = [len(x) for x in coords]

    # Create list of annotator labels
    ann_labels = np.array([0]*num_spots[0])
    for i in range(1, len(num_spots)):
        temp_labels = np.array([i]*num_spots[i])
        ann_labels = np.hstack((ann_labels, temp_labels))

    nodes = list(G_new.nodes)

    if exclude_last:
        for i in range(len(nodes)-1):
            G_new.nodes[i]['name'] = ann_labels[i]
    else:
        for i in range(len(nodes)):
            G_new.nodes[i]['name'] = ann_labels[i]

    return G_new


def label_graph_gt(G, detection_data, gt):
    """Labels the ground truth identity of each node in the graph -- intended for simulated data
    Parameters:
    ------------
    G : networkx graph
        Graph with edges indicating clusters of points assumed to be derived from the same ground
        truth detection
    detection_data : matrix
        Matrix with dimensions (number of clusters) x (number of algorithms) with value of 1 if an
        algorithm detected
        the cluster and 0 if it did not
    gt : array
        Array with length (number of cluster) with value of 1 if cluster is a true positive
        detection and 0 if it is a false positive

    Returns:
    ----------
    G_new : networkx graph
        Labeled graph
    """
    G_new = G.copy()

    num_annotators = np.shape(detection_data)[1]

    labels = []
    for i in range(num_annotators):
        detections = detection_data[:, i]

        for ii in range(len(detections)):
            if detections[ii] == 1:
                if gt[ii] == 1:
                    labels.append(1)
                if gt[ii] == 0:
                    labels.append(0)

    nodes = list(G.nodes)

    for i in range(len(nodes)-1):
        G_new.nodes[i]['name'] = labels[i]

    return G_new


def label_graph_prob(G, detection_data, p_matrix):
    """Labels the EM output probability of being a ground truth true detection for each cluster in
    the graph
    Parameters:
    ------------
    G : networkx graph
        Graph with edges indicating clusters of points assumed to be derived from the same ground
        truth detection
    detection_data : matrix
        Matrix with dimensions (number of clusters) x (number of algorithms) with value of 1 if an
        algorithm detected
        the cluster and 0 if it did not
    p_matrix : matrix
        Matrix with dimensions (number of clusters) x 2 where first column is the probability that
        a cluster is a true positive and second column is the probability that it is a false
        positive
    Returns:
    ----------
    G_new : networkx graph
        Labeled graph
    """
    G_new = G.copy()

    num_annotators = np.shape(detection_data)[1]

    labels = []
    for i in range(num_annotators):
        detections = detection_data[:, i]

        for ii in range(len(detections)):
            if detections[ii] == 1:
                labels.append(p_matrix[:, 0][ii])

    nodes = list(G.nodes)

    for i in range(len(nodes)-1):
        G_new.nodes[i]['name'] = labels[i]

    return G_new
