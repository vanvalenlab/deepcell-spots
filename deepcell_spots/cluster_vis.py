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


def label_graph_ann(G, coords_df, exclude_last=False):
    """Labels the annotator associated with each node in the graph

    Args:
        G (networkx.Graph): Graph with edges indicating clusters of points
            assumed to be derived from the same ground truth detection
        coords_df (DataFrame): Data frame with columns 'x' and 'y' which encode the
            spot locations and 'Algorithm' which encodes the algorithm that
            corresponds with that spot
        exclude_last (bool): Only set as True to exclude a point that has been
            included for the purpose of normalization

    Returns:
        networkx.Graph: Labeled graph
    """

    G_new = G.copy()
    algs = coords_df.Algorithm.unique()
    num_spots = [len(coords_df.loc[coords_df['Algorithm'] == alg]) for alg in algs]

    # Create list of annotator labels
    labels = []
    for i in range(len(num_spots)):
        labels.extend([i] * num_spots[i])

    nodes = list(G_new.nodes)

    if exclude_last:
        for i in range(len(nodes) - 1):
            G_new.nodes[i]['name'] = labels[i]
    else:
        for i in range(len(nodes)):
            G_new.nodes[i]['name'] = labels[i]

    return G_new


def label_graph_gt(G, detection_data, gt):
    """Labels the ground truth identity of each node in the graph.

    Intended for simulated data.

    Args:
        G (networkx.Graph): Graph with edges indicating clusters of points
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
