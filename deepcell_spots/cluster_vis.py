import numpy as np
import networkx as nx 
from scipy.spatial import distance
from itertools import combinations

# Functions for expectation maximization for spot detection


def jitter(coords,size):
    """ Add Gaussian noise to a list of coordinates for plotting when coordinates overlap.

    Parameters:
    ------------
    coords : matrix
        Matrix with dimensions (number of detections) x 2
    size : integer
        Standard deviation of the Gaussian noise distribution in pixels. 
    """
    result = []
    for item in coords:
        noise = np.random.normal(0,size)
        result.append(item+noise)

    return np.array(result)


def label_graph_ann(G, coords):
    """Labels the annotator associated with each node in the graph"""
    G_new = G.copy()
    num_spots = [len(x) for x in coords]

    # Create list of annotator labels
    ann_labels = np.array([0]*num_spots[0])
    for i in range(1,len(num_spots)):
        temp_labels = np.array([i]*num_spots[i])
        ann_labels = np.hstack((ann_labels,temp_labels))

    nodes = list(G_new.nodes)

    for i in range(len(nodes)-1):
        G_new.nodes[i]['name'] = ann_labels[i]

    return G_new

def label_graph_gt(G, data, gt):
    """Labels the ground truth identity of each node in the graph"""
    G_new = G.copy()
    
    num_annotators = np.shape(data)[1]

    labels = []
    for i in range(num_annotators):
        detections = data[:,i]

        for ii in range(len(detections)):
            if detections[ii] == 1:
                if gt[ii] == 'T':
                    labels.append(1)
                if gt[ii] == 'F':
                    labels.append(0)

    nodes = list(G.nodes)

    for i in range(len(nodes)-1):
        G_new.nodes[i]['name'] = labels[i]
 

    return G_new

def label_graph_prob(G, data, p_matrix):
    """Labels the EM output probability of being a ground truth true detection for each cluster in the graph"""
    G_new = G.copy()

    num_annotators = np.shape(data)[1]

    labels = []
    for i in range(num_annotators):
        detections = data[:,i]

        for ii in range(len(detections)):
            if detections[ii] == 1:
                labels.append(p_matrix[:,0][ii])

    nodes = list(G.nodes)

    for i in range(len(nodes)-1):
        G_new.nodes[i]['name'] = labels[i]
 

    return G_new

