import random
import numpy as np

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

def em_spot(data, tp_list, fp_list, prior, max_iter=10):
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
    
    likelihood_matrix = np.zeros((len(data), 2))

    for i in range(max_iter):
        # Caluclate the probability that each spot is a true detection or false detection
        likelihood_matrix = np.zeros((len(data), 2))

        for i in range(len(data)):
            likelihood_matrix[i] = norm_marg_likelihood(data[i], tp_list, fp_list, prior)

        # Calculate the expectation value for the number of TP/FN/FP/TN
        tp_matrix = np.zeros((np.shape(data)))
        for i in range(len(data)): 
            tp_matrix[i] = likelihood_matrix[i,0] * data[i]

        fn_matrix = np.zeros((np.shape(data)))
        for i in range(len(data)):
            fn_matrix[i] = likelihood_matrix[i,0] * (data[i] * -1 + 1)

        fp_matrix = np.zeros((np.shape(data)))
        for i in range(len(data)):  
            fp_matrix[i] = likelihood_matrix[i,1] * data[i]

        tn_matrix = np.zeros((np.shape(data)))
        for i in range(len(data)):
            tn_matrix[i] = likelihood_matrix[i,1] * (data[i] * -1 + 1)

        tp_sum_list = [sum(tp_matrix[:,i]) for i in range(np.shape(data)[1])]
        fn_sum_list = [sum(fn_matrix[:,i]) for i in range(np.shape(data)[1])]
        fp_sum_list = [sum(fp_matrix[:,i]) for i in range(np.shape(data)[1])]
        tn_sum_list = [sum(tn_matrix[:,i]) for i in range(np.shape(data)[1])]

        # Calculate the MLE estimate for the TPR/FPR
        tp_list = [tp_sum_list[i] / (tp_sum_list[i]+fn_sum_list[i]) for i in range(np.shape(data)[1])]
        fp_list = [fp_sum_list[i] / (fp_sum_list[i]+tn_sum_list[i]) for i in range(np.shape(data)[1])]


    likelihood_matrix = np.round(likelihood_matrix,2)

    return tp_list, fp_list, likelihood_matrix




