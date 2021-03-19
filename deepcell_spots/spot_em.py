import random
import numpy as np
import networkx as nx
from itertools import combinations
from scipy.spatial import distance
from copy import deepcopy
from deepcell_spots.cluster_vis import *

def calc_tpr_fpr(gt, data):
    """Calculate the true postivie rate and false positive rate for a pair of ground truth labels and detection data. 
    
    Parameters: 
    ------------
    gt : array-like
        Array of ground truth cluster labels. 'T' indicates a true detection and 'F' indicates a false detection. 
    data : array-like
        Array of detection data with same length . A value of 1 indicates a detected clsuter and a value of 0 indicates an undetected cluster. 
    
    Returns:
    ---------
    tpr : float
        Value for the true positive rate of an annotator. This is the probability that an annotator will detect a spot that is labeled as a ground truth true detection.
    fpr : flaot
        Value for the false positive rate of an annotator. This is the probability that an annotator will detect a spot that is labeled as a ground truth false detection. 
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for i in range(len(gt)):
        if gt[i] == 'T':
            if data[i] == 1:
                tp += 1
            elif data[i] == 0:
                fn += 1
        elif gt[i] == 'F':
            if data[i] == 1:
                fp += 1
            elif data[i] == 0:
                tn += 1

    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)

    return tpr, fpr

def det_likelihood(cluster_data, pr_list):
    """Calculate the likelihood that a cluster is a true positive or false positive. To calculate the likelihood of a true positive, pr_list should be a list of TPRs for all annotators. To calculate the likelihood of a cluster being a false positive, pr_list should be a list of FPRs for all annotators. 
    
    Returns a value for the likelihood that a cluster is either a true positive or a false positive
    
    Parameters:
    -----------
    cluster_data : array_like
        Array of detection labels for each annotator. Entry has value 1 if annotator detected the cluster, and entry has value 0 if annotator did not detect the cluster. 
    pr_list : array_like
        Array of true postive rates for each annotator if one wants to calculate the likelihood that the cluster is a true positive, or array of false positive rates for each annotator if one wants to calculate the likelihood that the cluster is a false positive. 

    Returns:
    ---------
    likelihood : float
        Value for the likelihood that a cluster is either a true positive or a false positive detection 
    
    """
    likelihood = 1

    for i in range(len(cluster_data)):
        if cluster_data[i] == 1:
            likelihood *= pr_list[i]
        elif cluster_data[i] == 0:
            likelihood *= (1-pr_list[i])
    return likelihood

def norm_marg_likelihood(cluster, tp_list, fp_list, prior):
    tp_likelihood = det_likelihood(cluster, tp_list)
    fp_likelihood = det_likelihood(cluster, fp_list)

    norm_tp_likelihood = tp_likelihood * prior / (tp_likelihood * prior + fp_likelihood * (1-prior))
    norm_fp_likelihood = fp_likelihood * (1-prior) / (tp_likelihood * prior + fp_likelihood * (1-prior))

    return norm_tp_likelihood, norm_fp_likelihood

def em_spot(cluster_matrix, tp_list, fp_list, prior=0.9, max_iter=10):
    """ Estimate the TPR/FPR and probability of true detection for various spot annotators using expectation maximization. 

    Returns the true positive rate and false positive rate for each annotator, and returns the probability that each spot is a true detection or false detection. 

    Parameters: 
    -----------
    data : matrix
        Matrix of detection labels for each spot for each annotator. Dimensions spots x annotators. A value of 1 indicates that the spot was detected by that annotator and a value of 0 indicates that the spot was not detected by that annotator. 
    tp_list : array-like
        Array of initial guesses for the true positive rates for each annotator. 
    fp_list : array-like
        Array of initial guesses for the false positive rates for each annotator. 
    prior : float
        Value for the prior probability that a spot is a true positive.
    max_iter : integer
        Value for the number of times the expectation maximization algorithm will iteratively calculate the MLE for the TPR and FPR of the annotators.

    Returns:
    -----------
    tp_list : array-like
        Array of final estimates for the true positive rates for each annotator. 
    fp_list : array-like
        Array of final estimates for the false postitive rates for each annotator. 
    likelihood_matrix : matrix
        Matrix of probabilities that each cluster is a true detection (column 0) or false detection (column 1). Dimensions spots x 2.
    """
    
    likelihood_matrix = np.zeros((len(cluster_matrix), 2))

    for i in range(max_iter):
        # Caluclate the probability that each spot is a true detection or false detection
        likelihood_matrix = np.zeros((len(cluster_matrix), 2))

        for i in range(len(cluster_matrix)):
            likelihood_matrix[i] = norm_marg_likelihood(cluster_matrix[i], tp_list, fp_list, prior)

        # Calculate the expectation value for the number of TP/FN/FP/TN
        tp_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)): 
            tp_matrix[i] = likelihood_matrix[i,0] * cluster_matrix[i]

        fn_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)):
            fn_matrix[i] = likelihood_matrix[i,0] * (cluster_matrix[i] * -1 + 1)

        fp_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)):  
            fp_matrix[i] = likelihood_matrix[i,1] * cluster_matrix[i]

        tn_matrix = np.zeros((np.shape(cluster_matrix)))
        for i in range(len(cluster_matrix)):
            tn_matrix[i] = likelihood_matrix[i,1] * (cluster_matrix[i] * -1 + 1)

        tp_sum_list = [sum(tp_matrix[:,i]) for i in range(np.shape(cluster_matrix)[1])]
        fn_sum_list = [sum(fn_matrix[:,i]) for i in range(np.shape(cluster_matrix)[1])]
        fp_sum_list = [sum(fp_matrix[:,i]) for i in range(np.shape(cluster_matrix)[1])]
        tn_sum_list = [sum(tn_matrix[:,i]) for i in range(np.shape(cluster_matrix)[1])]

        # Calculate the MLE estimate for the TPR/FPR
        tp_list = [tp_sum_list[i] / (tp_sum_list[i]+fn_sum_list[i]) for i in range(np.shape(cluster_matrix)[1])]
        fp_list = [fp_sum_list[i] / (fp_sum_list[i]+tn_sum_list[i]) for i in range(np.shape(cluster_matrix)[1])]


    likelihood_matrix = np.round(likelihood_matrix,2)

    return tp_list, fp_list, likelihood_matrix

def cluster_coords(all_coords,image_stack,threshold):
    # create one annotator data matrix from all images
    # first iteration out of loop
    coords = np.array([item[0] for item in all_coords])
    # adjacency matrix
    A = define_edges(coords, threshold)
    # create graph
    G=nx.from_numpy_matrix(A)
    # label each annotator on graph
    G_labeled = label_graph_ann(G, coords)
    # break up clusters with multiple spots from single annotator
    G_clean = check_spot_ann_num(G_labeled, coords)
    # calculate centroid of each cluster
    spot_centroids = cluster_centroids(G_clean, coords)

    # create annotator data matrix for first image
    cluster_matrix = ca_matrix(G_clean)
    ind_skipped = []
    num_spots_list = [len(cluster_matrix)]
    centroid_list = [spot_centroids]
    # iterate through images
    for i in range(1,len(all_coords[0])):
        len_list = np.array([len(item[i]) for item in all_coords])
        
        if 0 in len_list:
            ind_skipped.append(i)
            continue

        coords = np.array([item[i] for item in all_coords])
        A = define_edges(coords, threshold)
        G=nx.from_numpy_matrix(A)
        G_labeled = label_graph_ann(G, coords)
        G_clean = check_spot_ann_num(G_labeled, coords)

        spot_centroids = cluster_centroids(G_clean, coords)
        centroid_list.append(spot_centroids)

        temp_data = ca_matrix(G_clean)
        num_spots_list.append(len(temp_data))

        cluster_matrix = np.vstack((cluster_matrix, temp_data))

        image_stack_updated = np.delete(image_stack, ind_skipped, 0)
        image_stack_updated = np.expand_dims(image_stack_updated, axis=-1)

        all_coords_updated = deepcopy(all_coords)
        for i in range(len(all_coords_updated)):
            all_coords_updated = np.delete(all_coords[i], ind_skipped)

    return(cluster_matrix, centroid_list, all_coords_updated, image_stack_updated)

def running_total_spots(centroid_list):
    num_spots_list = [len(item) for item in centroid_list]
    running_total_spots = np.zeros(len(num_spots_list)+1)

    for i in range(len(num_spots_list)):
        running_total_spots[i+1] = sum(num_spots_list[:i+1])

    running_total_spots = running_total_spots.astype(int)

    return running_total_spots

def ca_to_adjacency_matrix(ca_matrix):
    num_clusters = np.shape(ca_matrix)[0]
    num_annnotators = np.shape(ca_matrix)[1]
    tot_det_list = [sum(ca_matrix[:,i]) for i in range(num_annnotators)]
    tot_num_detections = int(sum(tot_det_list))

    A = np.zeros((tot_num_detections, tot_num_detections))
    for i in range(num_clusters):
        det_list = np.ndarray.flatten(np.argwhere(ca_matrix[i] == 1))
        combos = list(combinations(det_list, 2))

        for ii in range(len(combos)):
            ann_index0 = combos[ii][0]
            ann_index1 = combos[ii][1]
            det_index0 = int(sum(tot_det_list[:ann_index0]) + sum(ca_matrix[:i,ann_index0]))
            det_index1 = int(sum(tot_det_list[:ann_index1]) + sum(ca_matrix[:i,ann_index1]))

            A[det_index0, det_index1] += 1
            A[det_index1, det_index0] += 1

    return A

def check_spot_ann_num(G, coords):
    """ Check that each annotator only has one spot per cluster, break up clusters that have multiple spots per annotator"""
    # Get all node labels
    node_labels = list(nx.get_node_attributes(G,'name').values())
    # Get all connected nodes = clusters
    clusters = list(nx.connected_components(G))
    flat_coords = np.vstack(coords)

    # Iterate through clusters 
    for i in range(len(clusters)):
        # Get points in a cluster
        cluster_pts = list(clusters[i])

        # Get labels for points in cluster 
        pt_labels = np.array(node_labels)[cluster_pts]
        
        # Check for repeated labels
        if len(pt_labels) != len(np.unique(pt_labels)):
            pt_labels = pt_labels.tolist()
            # Find which labels repeat
            repeats = set([x for x in pt_labels if pt_labels.count(x) > 1])
            repeats = list(repeats)
            # Calculate the centroid of the cluster 
            pt_locs = flat_coords[cluster_pts]
            centroid = [np.mean(pt_locs[:,0]), np.mean(pt_locs[:,1])]

            # Iterate through the repeated clusters 
            for ii in range(len(repeats)):
                # Get points with a particular repeated label
                repeat_pts = np.argwhere(np.array(pt_labels) == repeats[ii])
                repeat_pts = repeat_pts.tolist()

                # Find point that is the closest to the centroid of the cluster 
                pt_dists = [distance.euclidean(pt_locs[item],centroid) for item in repeat_pts]
                min_ind = np.argmin(pt_dists)

                # Remove the point that is the closest to the centroid of the cluster 
                repeat_pts.pop(min_ind)

                #need to break all edges that have repeating annotators 
                for item in repeat_pts:
                    repeat_node = cluster_pts[item[0]]
                    edges = list(G.edges(repeat_node))
                    for edge in edges:
                        G.remove_edge(*edge)
    return G

def ca_matrix(G):
    """Convert graph into cluster x annotators matrix where 0 mean annotator did not find a point in that cluster
    and 1 means that the annotator did find a point in that cluster"""
    clusters = list(nx.connected_components(G))
    node_labels = list(nx.get_node_attributes(G,'name').values())
    num_ann = len(np.unique(node_labels))

    ca_matrix = np.zeros((len(clusters), num_ann))
    for i in range(len(clusters)):
        cluster_pts = list(clusters[i])

        for item in cluster_pts:
            ca_matrix[i,node_labels[item]] += 1

    return ca_matrix

def define_edges(coords, threshold):
    """Defines that adjacency matrix for the multiple annotators, connecting points that are sufficiently close to one another
    It is assumed that these spots are derived from the same ground truth spot in the original image
    
    Parameters:
    -----------
    coords : array-like
        Array of coordinates from each annotator, length is equal to the number of annotators. Each item in the array is a matrix
        of detection locations with dimensions (number of detections)x2.
    threshold : float
        The distance in pixels. Detections closer than the threshold distance will be grouped into a "cluster" of detections, 
        assumed to be derived from the same ground truth detection. 

    Returns:
    ----------
    A : matrix
        Matrix of dimensions (number of detections) x (number of detections) defining edges of a graph clustering detections by 
        detections from different annotators derived from the same ground truth detection. A value of 1 denotes two connected nodes
        in the eventual graph and a value of 0 denotes disconnected nodes. 
    """
    # flatten detection coordinates into single 1d array
    all_coords = np.vstack(coords)
    num_spots = len(all_coords)

    A = np.zeros((num_spots, num_spots))
    for i in range(num_spots):
        for ii in range(i+1, num_spots):
            # calculate distance between points
            dist = np.linalg.norm(all_coords[i] - all_coords[ii])
            if dist < threshold:
                # define an edge if the detections are sufficiently close
                A[i][ii] += 1
                # symmetrical edge, because graph is non-directed
                A[ii][i] += 1

    return A


def cluster_centroids(G, coords):
    """ Calculate the location of the centroid of a cluster of detections.

    Returns a list of coordinates for the centroid of each cluster in an input graph.

    Parameters:
    -------------
    G : 
    """
    clusters = list(nx.connected_components(G))
    flat_coords = np.vstack(coords)

    centroid_list = []
    for i in range(len(clusters)):
        cluster_pts = list(clusters[i])
        pt_locs = flat_coords[cluster_pts]
        centroid = [np.mean(pt_locs[:,0]), np.mean(pt_locs[:,1])]
        centroid_list.append(centroid)

    return centroid_list
