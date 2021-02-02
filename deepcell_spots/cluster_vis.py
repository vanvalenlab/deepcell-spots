import numpy as np
import networkx as nx 
from scipy.spatial import distance
from itertools import combinations

# Functions for expectation maximization for spot detection
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

def make_data_stack(log_coords,dog_coords,plm_coords):
    # create one annotator data matrix from all images
    # first iteration out of loop
    log = log_coords[0]
    dog = dog_coords[0]
    plm = plm_coords[0]

    # cluster all detected spots in first image
    coords = np.array([plm, log, dog])
    # adjacency matrix
    A = define_edges(coords, 2)
    # create graph
    G=nx.from_numpy_matrix(A)
    # label each annotator on graph
    G_labeled = label_graph_ann(G, coords)
    # break up clusters with multiple spots from single annotator
    G_clean = check_spot_ann_num(G_labeled, coords)
    # calculate centroid of each cluster
    spot_centroids = cluster_centroids(G, coords)

    # create annotator data matrix for first image
    data_stack = ca_matrix(G_clean)
    ind_skipped = []
    num_spots_list = [len(data_stack)]
    centroid_list = [spot_centroids]
    # iterate through images
    for i in range(1,len(log_coords)):
        log = log_coords[i]
        dog = dog_coords[i]
        plm = plm_coords[i]

        if len(log) == 0 or len(dog) == 0 or len(plm) == 0:
            ind_skipped.append(i)
            continue

        coords = np.array([plm, log, dog])
        A = define_edges(coords, 2)
        G=nx.from_numpy_matrix(A)
        G_labeled = label_graph_ann(G, coords)
        G_clean = check_spot_ann_num(G_labeled, coords)

        spot_centroids = cluster_centroids(G, coords)
        centroid_list.append(spot_centroids)

        temp_data = ca_matrix(G_clean)
        num_spots_list.append(len(temp_data))

        data_stack = np.vstack((data_stack, temp_data))

    return(data_stack, centroid_list, num_spots_list, ind_skipped)

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