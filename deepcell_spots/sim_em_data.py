import numpy as np 
import random 


def gt_clusters(num_clusters, tp_ratio):
    """ Generate random simulated labels (true detection or false detection) for clusters, with a specified rate of true detections and false detections, tp_ratio.

    Returns a list of length num_clusters of cluster labels with value 'T' for a true detection and 'F' for a false detection.

    Parameters:
    -----------
    num_clusters : integer
        The number of cluster labels to be generated
    tp_ratio : float
        The average percentage of the detections that are true detections

    Returns:
    ----------
    gt : list
        List of random simulated cluster labels 'T' or 'F'

    """
    gt = []
    for i in range(num_clusters):
        rand = random.random()

        if rand < tp_ratio:
            gt.append('T')
        else:
            gt.append('F')
    return gt

def sim_detection(gt, tpr, fpr):
    """Simulates detection data for a set of ground truth cluster labels and an annotator with a specified TPR and FPR. 
    
    Returns an array of with same length as input gt, where 1 indicates the simulated annotator detected a cluster and 0 indicates an undetected cluster.
    
    Parameters:
    -------------
    gt : array-like
        Array of ground truth cluster labels. 'T' indicates a true detection and 'F' indicates a false detection. 
    tpr : float
        The true positive rate of the annotator. For a ground truth value of 'T', it is the probability that the function will output 1, indicating that the simulated annotator detected the true cluster. 
    fpr : float
        The false positive rate of the annotator. For a ground truth value of 'F', it is the probability that the funciton will output 1, indicating that the simulated annotator falsely detected the cluster.  
    
    Returns: 
    ----------
    det_list : array-like
        Array of detected cluster labels. A value of 1 indicates that a cluster was detected by the annotator, and 0 indicates that the cluster was not detected by the annotator. 
         """
    det_list = []
    for item in gt:
        rand = random.random()
        if item == 'T':
            if rand < tpr:
                det_list.append(1)
            else:
                det_list.append(0)
        elif item == 'F':
            if rand < fpr:
                det_list.append(1)
            else:
                det_list.append(0)

    return det_list

def sim_data(gt, tpr_list, fpr_list):
    """Simulate the detections of multiple annotators with different TPRs and FPRs on the same ground truth data. 
    
    Returns a matrix of simulated detection data with dimensions clusters x annotators. 
    
    Parameters:
    ------------
    gt : array-like
        Array of ground truth cluster labels. 'T' indicates a true detection and 'F' indicates a false detection. 
    tpr_list : array-like
        Array of TPR values for each annotator. For a ground truth value of 'T', the TPR is the probability that the function will output 1, indicating that the simulated annotator detected the true cluster. 
    fpr_list : array-like
        Array of FPR values for each annotator. For a ground truth value of 'F', the FPR is the probability that the funciton will output 1, indicating that the simulated annotator falsely detected the cluster.  

    Returns:
    --------
    data_array : matrix
        Matrix of simulated detection data with dimensions clusters x annotators. A value of 1 indicates a detected clsuter and a value of 0 indicates an undetected cluster. 
    """

    data_list = []
    for i in range(len(tpr_list)):
        data_list.append(sim_detection(gt, tpr_list[i], fpr_list[i]))

    data_array = np.array(data_list).T

    return data_array

def percent_correct(gt, p_matrix):
    num_correct = 0

    for i in range(len(gt)):
        label = np.round(p_matrix[i,0])

        if gt[i] == 'T' and label == 1:
            num_correct += 1
        elif gt[i] == 'F' and label == 0:
            num_correct += 1

    return num_correct / len(gt)