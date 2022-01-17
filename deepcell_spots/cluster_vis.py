# Copyright 2019-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-spots/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Visualization tools for spot expectation maximization"""

from itertools import combinations

import numpy as np


def jitter(coords, size):
    """Add Gaussian noise to a list of coordinates for plotting when coordinates overlap.

    Args:
        coords (matrix): Matrix with dimensions (number of detections) x 2
        size (int): Standard deviation of the Gaussian noise distribution in pixels.

    Returns:
        array: Coords with noise added to locations
    """
    result = []
    for item in coords:
        noise = np.random.normal(0, size)
        result.append(item + noise)

    return np.array(result)


def ca_to_adjacency_matrix(ca_matrix):
    num_clusters = np.shape(ca_matrix)[0]
    num_annnotators = np.shape(ca_matrix)[1]
    tot_det_list = [sum(ca_matrix[:, i]) for i in range(num_annnotators)]
    tot_num_detections = int(sum(tot_det_list))

    A = np.zeros((tot_num_detections, tot_num_detections))
    for i in range(num_clusters):
        det_list = np.ndarray.flatten(np.argwhere(ca_matrix[i] == 1))
        combos = list(combinations(det_list, 2))

        for ii in range(len(combos)):
            ann_index0 = combos[ii][0]
            ann_index1 = combos[ii][1]
            det_index0 = int(
                sum(tot_det_list[:ann_index0]) + sum(ca_matrix[:i, ann_index0]))
            det_index1 = int(
                sum(tot_det_list[:ann_index1]) + sum(ca_matrix[:i, ann_index1]))

            A[det_index0, det_index1] += 1
            A[det_index1, det_index0] += 1

    return A


def label_graph_ann(G, coords, exclude_last=False):
    """Labels the annotator associated with each node in the graph

    Args:
        G (networkx.Graph): Graph with edges indicating clusters of points
        assumed to be derived from the same ground truth detection
    coords (numpy.array): 2d-array of detected point locations for each
        classical algorithm used
    exclude_last (bool): Only set as True to exclude a point that has been
        included for the purpose of normalization

    Returns:
        networkx.Graph: Labeled graph
    """

    G_new = G.copy()
    num_spots = [len(x) for x in coords]

    # Create list of annotator labels
    ann_labels = np.array([0] * num_spots[0])
    for i in range(1, len(num_spots)):
        temp_labels = np.array([i] * num_spots[i])
        ann_labels = np.hstack((ann_labels, temp_labels))

    nodes = list(G_new.nodes)

    if exclude_last:
        for i in range(len(nodes) - 1):
            G_new.nodes[i]['name'] = ann_labels[i]
    else:
        for i in range(len(nodes)):
            G_new.nodes[i]['name'] = ann_labels[i]

    return G_new


def label_graph_gt(G, detection_data, gt):

    """Labels the ground truth identity of each node in the graph.

    Intended for simulated data.

    Args:
        G (networkx graph): Graph with edges indicating clusters of points
            assumed to be derived from the same ground truth detection
        detection_data (numpy.array): Matrix with dimensions (number of clusters) x
            (number of algorithms) with value of 1 if an algorithm detected
            the cluster and 0 if it did not.
        gt (numpy.array): Array with length (number of cluster) with value of 1 if
            cluster is a true positive detection and 0 if it is a false positive.

    Returns:
        networkx.Graph: Labeled graph
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

    for i in range(len(nodes) - 1):
        G_new.nodes[i]['name'] = labels[i]

    return G_new


def label_graph_prob(G, detection_data, p_matrix):

    """Labels the EM output probability of being a ground truth true detection
    for each cluster in the graph.

    Args:
        G (networkx.Graph): Graph with edges indicating clusters of points
            assumed to be derived from the same ground truth detection
        detection_data (numpy.array): Matrix with dimensions (number of
            clusters) x (number of algorithms) with value of 1 if an algorithm
            detected the cluster and 0 if it did not.
        p_matrix (matrix): Matrix with dimensions (number of clusters) x 2
            where first column is the probability that a cluster is a true
            positive and second column is the probability that it is a
            false positive.

    Returns:
        networkx.Graph: Labeled graph
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

    for i in range(len(nodes) - 1):
        G_new.nodes[i]['name'] = labels[i]

    return G_new
