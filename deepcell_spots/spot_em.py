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

"""Expectation maximization functions for spot detection"""

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm

from deepcell_spots.cluster_vis import label_graph_ann

# supress a chained assignment warning
pd.set_option('chained_assignment', None)


def calc_tpr_fpr(gt, data):
    """Calculate the true postivie rate and false positive rate for a pair of
    ground truth labels and detection data.

    Args:
        gt (array): Array of ground truth cluster labels. A value of 1 indicates
            a true detection and a value of 0 indicates a false detection.
    data (array): Array of detection data with same length.
        A value of 1 indicates a detected cluster and a value of 0 indicates
        an undetected cluster.

    Returns:
        float: Value for the true positive rate of an annotator.
            This is the probability that an annotator will detect a spot that
            is labeled as a ground truth true detection.
        float: Value for the false positive rate of an annotator.
            This is the probability that an annotator will detect a spot that
            is labeled as a ground truth false detection.
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for i in range(len(gt)):
        if gt[i] == 1:
            if data[i] == 1:
                tp += 1
            elif data[i] == 0:
                fn += 1
        elif gt[i] == 0:
            if data[i] == 1:
                fp += 1
            elif data[i] == 0:
                tn += 1

    tpr = tp / (tp + fn)

    fpr = fp / (fp + tn)

    return tpr, fpr


def det_likelihood(cluster_data, pr_list):
    """Calculate the likelihood that a cluster is a true positive or false
    positive. To calculate the likelihood of a true positive, pr_list should
    be a list of TPRs for all annotators. To calculate the likelihood of a
    cluster being a false positive, pr_list should be a list of FPRs for all
    annotators.

    Returns a value for the likelihood that a cluster is either a true
    positive or a false positive.

    Args:
        cluster_data (array): Array of detection labels for each annotator.
            Entry has value 1 if annotator detected the cluster, and entry has
            value 0 if annotator did not detect the cluster.
        pr_list (array): Array of true postive rates for each annotator if one
            wants to calculate the likelihood that the cluster is a true
            positive, or array of false positive rates for each annotator if
            one wants to calculate the likelihood that the cluster is a false
            positive.

    Returns:
        float: Value for the likelihood that a cluster is either a true positive
            or a false positive detection.
    """
    likelihood = 1  # init likelihood

    for i in range(len(cluster_data)):
        if cluster_data[i] == 1:
            likelihood *= pr_list[i]
        elif cluster_data[i] == 0:
            likelihood *= (1 - pr_list[i])
    return likelihood


def norm_marg_likelihood(cluster_data, tp_list, fp_list, prior):

    """Calculates the normalized marginal likelihood that each cluster is a
    true positive or a false positive.

    Args:
        cluster_data (array): Array of detection labels for each annotator.
            Entry has value 1 if annotator detected the cluster, and entry has
            value 0 if annotator did not detect the cluster.
        tp_list (array): Array of true postive rates for each annotator.
        fp_list (array): Array of false postive rates for each annotator.

    Returns:
        float: Value for the normalized marginal likelihood that a cluster is
            either a true positive detection
        float: Value for the normalized marginal likelihood that a cluster is
            either a false positive detection
    """
    tp_likelihood = det_likelihood(cluster_data, tp_list)
    fp_likelihood = det_likelihood(cluster_data, fp_list)

    norm_tp_likelihood = tp_likelihood * prior / \
        (tp_likelihood * prior + fp_likelihood * (1 - prior))
    norm_fp_likelihood = fp_likelihood * \
        (1 - prior) / (tp_likelihood * prior + fp_likelihood * (1 - prior))

    return norm_tp_likelihood, norm_fp_likelihood


def define_edges(coords_df, threshold):

    """Defines that adjacency matrix for the multiple annotators, connecting
    points that are sufficiently close to one another. It is assumed that these
    spots are derived from the same ground truth spot in the original image.

    Args:
        coords (DataFrame): Data frame with columns 'x' and 'y' which encode the
            spot locations and 'Algorithm' which encodes the algorithm that
            corresponds with that spot
        threshold (float): The distance in pixels. Detections closer than the
            threshold distance will be grouped into a "cluster" of detections,
            assumed to be derived from the same ground truth detection.

    Returns:
        numpy.array: Matrix of dimensions (number of detections) x (number of
            detections) defining edges of a graph clustering detections by
            detections from different annotators derived from the same ground
            truth detection. A value of 1 denotes two connected nodes in the
            eventual graph and a value of 0 denotes disconnected nodes.
    """
    if not all(col in coords_df.columns for col in ['x', 'y', 'Algorithm']):
        raise NameError('coords_df must be a Pandas dataframe with columns'
                        '\'x\', \'y\', and \'Algorithm\'')

    all_coords = np.array([coords_df['x'], coords_df['y']]).T
    num_spots = len(all_coords)

    A = np.zeros((num_spots, num_spots))
    for i in range(num_spots):
        alg = coords_df['Algorithm'][i]

        for ii in range(i + 1, num_spots):
            # skip iteration if same algorithm
            temp_alg = coords_df['Algorithm'][ii]
            if alg == temp_alg:
                continue
            # calculate distance between points
            dist = np.linalg.norm(all_coords[i] - all_coords[ii])
            if dist < threshold:
                # define an edge if the detections are sufficiently close
                A[i][ii] += 1
                # symmetrical edge, because graph is non-directed
                A[ii][i] += 1

    return A


def em_spot(cluster_matrix, tp_list, fp_list, prior=0.9, max_iter=10):

    """Estimate the TPR/FPR and probability of true detection for various spot
    annotators using expectation maximization.

    Returns the true positive rate and false positive rate for each annotator,
    and returns the probability that each spot is a true detection or false
    detection.

    Args:
        cluster_matrix (matrix): Matrix of detection labels for each spot for
            each annotator. Dimensions spots x annotators. A value of 1
            indicates that the spot was detected by that annotator and a value
            of 0 indicates that the spot was not detected by that annotator.
        tp_list (array): Array of initial guesses for the true positive rates
            for each annotator.
        fp_list (array): Array of initial guesses for the false positive rates
            for each annotator.
        prior (float): Value for the prior probability that a spot is a true
            positive.
        max_iter (int): Value for the number of times the expectation
            maximization algorithm will iteratively calculate the MLE for the
            TPR and FPR of the annotators.

    Returns:
        tp_list (array): Array of final estimates for the true positive rates
            for each annotator.
        fp_list (array): Array of final estimates for the false postitive rates
            for each annotator.
        likelihood_matrix (matrix): Matrix of probabilities that each cluster
            is a true detection (column 0) or false detection (column 1).
            Dimensions spots x 2.
    """

    likelihood_matrix = np.zeros((len(cluster_matrix), 2))

    for i in range(max_iter):
        # Caluclate the probability that each spot is a true detection or false detection
        likelihood_matrix = np.zeros((len(cluster_matrix), 2))

        for i in range(len(cluster_matrix)):
            likelihood_matrix[i] = norm_marg_likelihood(
                cluster_matrix[i], tp_list, fp_list, prior)

        # Calculate the expectation value for the number of TP/FN/FP/TN
        tp_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)):
            tp_matrix[i] = likelihood_matrix[i, 0] * cluster_matrix[i]

        fn_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)):
            fn_matrix[i] = likelihood_matrix[i, 0] * \
                (cluster_matrix[i] * -1 + 1)

        fp_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)):
            fp_matrix[i] = likelihood_matrix[i, 1] * cluster_matrix[i]

        tn_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)):
            tn_matrix[i] = likelihood_matrix[i, 1] * \
                (cluster_matrix[i] * -1 + 1)

        tp_sum_list = [sum(tp_matrix[:, i])
                       for i in range(np.shape(cluster_matrix)[1])]
        fn_sum_list = [sum(fn_matrix[:, i])
                       for i in range(np.shape(cluster_matrix)[1])]
        fp_sum_list = [sum(fp_matrix[:, i])
                       for i in range(np.shape(cluster_matrix)[1])]
        tn_sum_list = [sum(tn_matrix[:, i])
                       for i in range(np.shape(cluster_matrix)[1])]

        # Calculate the MLE estimate for the TPR/FPR
        tp_list = [tp_sum_list[i] / (tp_sum_list[i] + fn_sum_list[i])
                   for i in range(np.shape(cluster_matrix)[1])]
        fp_list = [fp_sum_list[i] / (fp_sum_list[i] + tn_sum_list[i])
                   for i in range(np.shape(cluster_matrix)[1])]

    likelihood_matrix = np.round(likelihood_matrix, 6)

    return tp_list, fp_list, likelihood_matrix


def load_coords(coords):
    coords_df = pd.DataFrame(columns=['Algorithm', 'Image', 'x', 'y', 'Cluster'])

    for key in coords.keys():
        one_alg_coords = coords[key]

        for i in range(len(one_alg_coords)):
            num_spots = len(one_alg_coords[i])

            temp_df = pd.DataFrame(columns=['Algorithm', 'Image', 'x', 'y', 'Cluster'])
            temp_df['Algorithm'] = [key] * num_spots
            temp_df['Image'] = [i] * num_spots
            temp_df['x'] = one_alg_coords[i][:, 1]
            temp_df['y'] = one_alg_coords[i][:, 0]
            temp_df['Cluster'] = [0] * num_spots

            coords_df = pd.concat([coords_df, temp_df])

    coords_df = coords_df.reset_index(drop=True)

    return coords_df


def cluster_coords(coords_df, threshold):
    for i in tqdm(range(len(coords_df['Image'].unique()))):
        image_df = coords_df.loc[coords_df['Image'] == i]
        image_df = image_df.reset_index(drop=True)

        if len(image_df) == 0:
            continue

        A = define_edges(image_df, threshold=threshold)

        G = nx.from_numpy_matrix(A)
        G_labeled = label_graph_ann(G, image_df)

        clusters = list(nx.connected_components(G_labeled))

        cluster_labels = np.zeros(len(image_df))
        for ii in range(len(clusters)):
            for iii in range(len(clusters[ii])):
                cluster_labels[list(clusters[ii])[iii]] = ii

        image_df['Cluster'] = cluster_labels
        image_df = image_df.sort_values(by=['Cluster'])
        image_df = image_df.reset_index(drop=True)

        for item in image_df.Cluster.unique():
            # slice data frame by cluster
            cluster_df = image_df.loc[image_df['Cluster'] == item]
            cluster_df = cluster_df.sort_values(by=['Algorithm'])
            cluster_df = cluster_df.reset_index(drop=True)

            # check if more than one detection per alg in each cluster
            if any(cluster_df['Algorithm'].value_counts() > 1):
                # find algorithms with more than one detection
                count_dict = dict(cluster_df['Algorithm'].value_counts())
                multiple_keys = [key for key in count_dict.keys() if count_dict[key] > 1]

                # get centroid of cluster
                centroid_x = np.mean(cluster_df['x'])
                centroid_y = np.mean(cluster_df['y'])
                centroid = np.array([centroid_x, centroid_y])

                # calculate distance to centroid
                distance_list = []
                for i in range(len(cluster_df)):
                    distance_list.append(np.linalg.norm(centroid - np.array([cluster_df['x'][i],
                                                                             cluster_df['y'][i]])))
                cluster_df['Distance'] = distance_list

                highest_cluster = max(image_df['Cluster']) + 1
                for alg in multiple_keys:
                    # slice data frame by algorithm
                    alg_df = cluster_df.loc[cluster_df['Algorithm'] == alg]
                    alg_df = alg_df.reset_index(drop=True)
                    new_cluster_list = []

                    # assign spots far from centroid to new cluster
                    min_dist = min(alg_df['Distance'])
                    for ii in range(len(alg_df)):
                        if alg_df['Distance'][ii] > min_dist:
                            new_cluster_list.append(highest_cluster)
                            highest_cluster += 1
                        else:
                            new_cluster_list.append(item)

                    # replace values in cluster data frame
                    alg_df['Cluster'] = new_cluster_list
                    cluster_df = cluster_df.drop(cluster_df[cluster_df.Algorithm == alg].index)
                    cluster_df = pd.concat([cluster_df, alg_df])

                # replace values in image data frame
                del cluster_df['Distance']
                image_df = image_df.drop(image_df[image_df.Cluster == item].index)
                image_df = pd.concat([image_df, cluster_df])

        # re-sort data frame
        image_df = image_df.sort_values(by=['Cluster'])
        image_df = image_df.reset_index(drop=True)

        coords_df = coords_df.drop(coords_df[coords_df.Image == i].index)
        coords_df = pd.concat([coords_df, image_df])
        coords_df = coords_df.reset_index(drop=True)

    coords_df = coords_df.sort_values(by=['Image', 'Cluster'])
    coords_df = coords_df.reset_index(drop=True)
    return coords_df


def predict_cluster_probabilities(coords_df, tpr_dict, fpr_dict, prior=0.9, max_iter=10):
    lookup_dict = {i: item for i, item in enumerate(coords_df.Algorithm.unique())}
    num_algorithms = len(lookup_dict.keys())

    images = coords_df.Image.unique()
    num_clusters = [max(coords_df.loc[coords_df.Image == im].Cluster) + 1 for im in images]
    total_clusters = int(sum(num_clusters))

    copy_df = coords_df.copy()
    cluster_counter = []
    for im in copy_df.Image.unique():
        image_df = copy_df.loc[copy_df.Image == im]

        for c in range(len(image_df.Cluster.unique())):
            cluster_df = image_df.loc[image_df.Cluster == c]
            if len(cluster_counter) == 0:
                cluster_counter.extend([0] * len(cluster_df))
            else:
                cluster_counter.extend([cluster_counter[-1]+1] * len(cluster_df))

    copy_df['Cluster'] = cluster_counter

    cluster_matrix = np.zeros((total_clusters, num_algorithms))
    for i in tqdm(range(total_clusters)):
        algs = list(copy_df.loc[copy_df.Cluster == i]['Algorithm'])
        for ii in range(num_algorithms):
            if lookup_dict[ii] in algs:
                cluster_matrix[i, ii] += 1

    tp_guess = np.zeros((len(lookup_dict.keys())))
    fp_guess = np.zeros((len(lookup_dict.keys())))

    for key in lookup_dict.keys():
        tp_guess[key] = tpr_dict[lookup_dict[key]]
        fp_guess[key] = fpr_dict[lookup_dict[key]]

    tp_final_all, fp_final_all, p_matrix_all = em_spot(cluster_matrix,
                                                       tp_guess,
                                                       fp_guess,
                                                       prior,
                                                       max_iter)

    probability_list = []
    centroid_x_list = []
    centroid_y_list = []
    for c in range(len(copy_df.Cluster.unique())):
        cluster_df = copy_df.loc[copy_df.Cluster == c]
        probability_list.extend([p_matrix_all[c, 0]] * len(cluster_df))

        # get centroid of cluster
        centroid_x = np.mean(cluster_df['x'])
        centroid_y = np.mean(cluster_df['y'])
        centroid_x_list.extend([centroid_x] * len(cluster_df))
        centroid_y_list.extend([centroid_y] * len(cluster_df))

    copy_df['Probability'] = probability_list
    copy_df['Centroid_x'] = centroid_x_list
    copy_df['Centroid_y'] = centroid_y_list
    return copy_df
