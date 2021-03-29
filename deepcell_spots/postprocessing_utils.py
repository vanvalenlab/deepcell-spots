import numpy as np

from skimage.feature import peak_local_max
from skimage import measure

from deepcell_spots.point_metrics import stats_points


# functions for processing the neural network output into a final list of predicted coordinates

def y_annotations_to_point_list(y_pred, ind, threshold):
    """ Convert raw prediction to a predicted point list
    
    Args:
    y_pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for image #ind in the batch
    where annot_type = 0 or 1: 0 - contains_dot, 1 - offset matrices
    
    ind: the index of the image in the batch for which to convert the annotations
    
    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered containing a spot center,
    and their corresponding regression values will be used to create a final spot position prediction which will
    be added to the output spot center coordinates list.
    
    Returns:
    A list of spot center coordinates of the format [[y0, x0], [y1, x1],...]
    """

    contains_dot = y_pred[1][ind,...,1] > threshold
    delta_y = y_pred[0][ind,...,0]
    delta_x = y_pred[0][ind,...,1]

    dot_pixel_inds = np.argwhere(contains_dot)
    dot_centers = np.array([[y_ind+delta_y[y_ind, x_ind],x_ind+delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])
    return dot_centers

def y_annotations_to_point_list2(y_pred, ind, threshold):
    # make final decision to be: classification of pixel as containing dot > threshold AND
    # center regression is contained in the pixel
    contains_dot = y_pred[1][ind,...,1] > threshold
    delta_y = y_pred[0][ind,...,0]
    delta_x = y_pred[0][ind,...,1]
    contains_its_regression = (abs(delta_x)<=0.5) & (abs(delta_y)<=0.5)
    
    final_dot_detection = contains_dot & contains_its_regression

    dot_pixel_inds = np.argwhere(final_dot_detection)
    dot_centers = np.array([[y_ind+delta_y[y_ind, x_ind],x_ind+delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])
    return dot_centers

def y_annotations_to_point_list_restrictive(y_pred, ind, threshold):
    # make final decision to be: classification of pixel as containing dot > threshold AND
    # center regression is contained in the pixel
    contains_dot = y_pred[1][ind,...,1] > threshold
    delta_y = y_pred[0][ind,...,0]
    delta_x = y_pred[0][ind,...,1]
    contains_its_regression = (abs(delta_x)<=0.5) & (abs(delta_y)<=0.5)
    
    final_dot_detection = contains_dot & contains_its_regression

    dot_pixel_inds = np.argwhere(final_dot_detection)
    dot_centers = np.array([[y_ind+delta_y[y_ind, x_ind],x_ind+delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])
    return dot_centers


def y_annotations_to_point_list_with_source(y_pred, ind, threshold):
    """ Convert raw prediction to a predicted point list
    
    Args:
    y_pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for image #ind in the batch
    where annot_type = 0 or 1: 0 - contains_dot, 1 - offset matrices
    
    ind: the index of the image in the batch for which to convert the annotations
    
    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered containing a spot center,
    and their corresponding regression values will be used to create a final spot position prediction which will
    be added to the output spot center coordinates list.
    
    Returns:
    A list of spot center coordinates of the format [[y0, x0], [y1, x1],...]
    """

    contains_dot = y_pred[1][ind,...,1] > threshold
    delta_y = y_pred[0][ind,...,0]
    delta_x = y_pred[0][ind,...,1]

    dot_pixel_inds = np.argwhere(contains_dot)
    dot_centers = np.array([[y_ind+delta_y[y_ind, x_ind],x_ind+delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])
    return dot_centers, dot_pixel_inds



def y_annotations_to_point_list_max(y_pred, ind, threshold=0.95, min_distance=2):
    """ Convert raw prediction to a predicted point list
    
    Args:
    y_pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for image #ind in the batch
    where annot_type = 0 or 1: 0 - contains_dot (from classification head), 1 - offset matrices (from regression head)
    
    ind: the index of the image in the batch for which to convert the annotations
    
    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered containing a spot center,
    and their corresponding regression values will be used to create a final spot position prediction which will
    be added to the output spot center coordinates list.

    min_distance: the minimum distance between detected spots in pixels
    
    Returns:
    A list of spot center coordinates of the format [[y0, x0], [y1, x1],...]
    """
    dot_pixel_inds = peak_local_max(y_pred[1][ind,...,1], min_distance=min_distance, threshold_abs=threshold)

    delta_y = y_pred[0][ind,...,0]
    delta_x = y_pred[0][ind,...,1]
  
    dot_centers = np.array([[y_ind+delta_y[y_ind, x_ind],x_ind+delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])
    return dot_centers


def y_annotations_to_point_list_cc(y_pred, ind, threshold=0.8):
    # make final decision to be: average regression over each connected component of above detection threshold pixels

    delta_y = y_pred[0][ind,...,0]
    delta_x = y_pred[0][ind,...,1]
    
    blobs = contains_dot = y_pred[1][ind,...,1] > threshold
    label_image = measure.label(blobs, background=0)
    rp = measure.regionprops(label_image)

    dot_centers = []
    for region in rp:
        region_pixel_inds = region.coords
        reg_pred = [[y_ind+delta_y[y_ind, x_ind],x_ind+delta_x[y_ind, x_ind]] for y_ind, x_ind in region_pixel_inds]
        dot_centers.append(np.mean(reg_pred, axis=0))

    dot_centers = np.array(dot_centers)
    return dot_centers



def get_mean_stats(decision_function, y_test,y_pred,threshold=0.8,d_thresh=3):
    # decision_function: a postprocessing function with inputs y_pred, ind, threshold
    # this is a function that performs postprocessing on the output of the neural net for the image #ind in the batch,
    # and returns a decision for the list of coordinates of spot centers
    # d_thresh = 2 # distance in pixels for precision quantification
    # threshold = 0.95 # threshold for classification
    n_test = len(y_test) # number of test images

    d_md_list = [None]*n_test
    precision_list = [None]*n_test
    recall_list = [None]*n_test
    F1_list = [None]*n_test


    for ind in range(n_test): # loop over test images
        #y_pred_ind = y_annotations_to_point_list(y_pred_test, ind, threshold)
        y_pred_ind = decision_function(y_pred, ind, threshold)
        s = stats_points(y_test[ind], y_pred_ind, threshold=d_thresh)
        d_md_list[ind] = s['d_md']
        precision_list[ind] = s['precision']
        recall_list[ind] = s['recall']
        F1_list[ind] = s['Fmeasure']


    d_md = np.mean(d_md_list)
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    F1 = np.mean(F1_list)
    
    return (d_md, precision, recall, F1)

def model_benchmarking(pred,coords,threshold,min_distance):
    """Calculates the precision, recall, F1 score, Jacard Index, root mean square error, and sum of min distances for stack of predictions"

    Args:
    pred: a batch of predictions, of the format: y_pred[annot_type][ind] is an annotation for image #ind in the batch
    where annot_type = 0 or 1: 0 - contains_dot (from classification head), 1 - offset matrices (from regression head)
    coords: nested list of coordinate locations for ground truth spots from a single annotator
    threshold: a number in [0, 1]. Pixels with classification score > threshold are considered containing a spot center,
    and their corresponding regression values will be used to create a final spot position prediction which will
    be added to the output spot center coordinates list.
    min_distance: the minimum distance between detected spots in pixels

    Returns:
    precision: list of values for the precision of the predicted spot numbers in each image
    recall: list of values for the recall of the predicted spot numbers in each image
    f1: list of values for the f1 score of the predicted spot numbers in each image
    jac: list of values for the jacard index of the predicted spot numbers in each image
    rmse: list of values for the root mean square error of the spot locations in each image
    dmd: list of values for the sum of min distance of the spot locations in each image 

    """
    precision = []
    recall = []
    f1 = []
    jac = []
    rmse = []
    dmd = []

    for i in range(len(pred[0])):
        points_list2 = y_annotations_to_point_list_max(pred, i, threshold,min_distance)
        stats_dict = stats_points(coords[i],points_list2,1,match_points_function=match_points_mutual_nearest_neighbor)
        precision.append(stats_dict['precision'])
        recall.append(stats_dict['recall'])
        f1.append(stats_dict['F1'])
        jac.append(stats_dict['JAC'])
        rmse.append(stats_dict['RMSE'])
        dmd.append(stats_dict['d_md'])
        
    return(precision,recall,f1,jac,rmse,dmd)


def y_classification_to_point_list_max(y_pred, ind, threshold=0.8, min_distance=2):
    # make final decision to be: center of pixels that are local maxima of the classification detections
    # (use classification only to make final decision here, without using the regressed dy, dx)
    dot_pixel_inds = peak_local_max(y_pred[1][ind,...,1], min_distance=min_distance, threshold_abs=threshold)
 
    return dot_pixel_inds

