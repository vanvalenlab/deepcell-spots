# Copyright 2019-2023 The Van Valen Lab at the California Institute of
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

"""Data simulators for spot images for benchmarking deep learning model
     and annotator detections for benchmarking EM algorithm"""

import random

import numpy as np
import skimage


def sim_gt_clusters(num_clusters, tp_ratio):
    """Generate random simulated labels (true detection or false detection)
    for clusters, with a specified rate of true detections and false detections,
    `tp_ratio`.

    Returns a list of length `num_clusters` of cluster labels with value 1 for a
    true detection and 0 for a false detection.

    Args:
        num_clusters (int): The number of cluster labels to be generated.
        tp_ratio (float): The average percentage of the detections that are
            true detections.

    Returns:
        list: List of random simulated cluster labels 1 or 0.
    """

    assert tp_ratio >= 0 and tp_ratio <= 1, "TP ratio must be between 0 and 1"

    gt = []
    for i in range(num_clusters):
        rand = random.random()

        if rand < tp_ratio:
            gt.append(1)
        else:
            gt.append(0)
    return gt


def sim_detections(gt, tpr, fpr):
    """Simulates detection data for a set of ground truth cluster labels and an
    annotator with a specified TPR and FPR.

    Returns an array of with same length as input `gt`, where 1 indicates the
    simulated annotator detected a cluster and 0 indicates an undetected
    cluster.

    Args:
        gt (array): Array of ground truth cluster labels. 1 indicates a true
            detection and 0 indicates a false detection.
        tpr (float): The true positive rate of the annotator. For a ground
            truth value of 1, it is the probability that the function will
            output 1, indicating that the simulated annotator detected the
            true cluster.
        fpr (float): The false positive rate of the annotator. For a ground
            truth value of 0, it is the probability that the funciton will
            output 1, indicating that the simulated annotator falsely detected
            the cluster.

    Returns:
        array: Array of detected cluster labels. A value of 1 indicates that
        a cluster was detected by the annotator, and 0 indicates that the
        cluster was not detected by the annotator.
    """

    assert tpr >= 0 and tpr <= 1, "TPR must be between 0 and 1"
    assert fpr >= 0 and fpr <= 1, "FPR must be between 0 and 1"

    det_list = []
    for item in gt:
        rand = random.random()
        if item == 1:
            if rand < tpr:
                det_list.append(1)
            else:
                det_list.append(0)
        elif item == 0:
            if rand < fpr:
                det_list.append(1)
            else:
                det_list.append(0)

    return det_list


def sim_annotators(gt, tpr_list, fpr_list):
    """Simulate the detections of multiple annotators with different TPRs and
    FPRs on the same ground truth data.

    Returns a matrix of simulated detection data with dimensions clusters x
    annotators.

    Args:
        gt (array): Array of ground truth cluster labels. 1 indicates a true
            detection and 0 indicates a false detection.
        tpr_list (array): Array of TPR values for each annotator. For a ground
            truth value of 1, the TPR is the probability that the function
            will output 1, indicating that the simulated annotator detected
            the true cluster.
        fpr_list (array): Array of FPR values for each annotator. For a ground
            truth value of 0, the FPR is the probability that the funciton will
            output 1, indicating that the simulated annotator falsely detected
            the cluster.

    Returns:
        numpy.array: Matrix of simulated detection data with dimensions
        clusters x annotators. A value of 1 indicates a detected cluster
        and a value of 0 indicates an undetected cluster.
    """

    assert type(tpr_list) == list or type(
        tpr_list) == np.ndarray, "tpr_list must be a list or an array"
    assert type(fpr_list) == list or type(
        fpr_list) == np.ndarray, "fpr_list must be a list or an array"

    assert len(tpr_list) == len(
        fpr_list), "Length of TPR list and FPR list must be the same"

    data_list = []
    for i in range(len(tpr_list)):
        data_list.append(sim_detections(gt, tpr_list[i], fpr_list[i]))

    data_matrix = np.array(data_list).T

    return data_matrix


def percent_correct(gt, data_array):
    """Calculates the percent of detections correctly labeled.

    Returns a value from 0 to 1 indicating the fraction of detections correctly
    labeled.

    Args:
        gt (array): Array of ground truth cluster labels. 1 indicates a true
            detection and 0 indicates a false detection.
        data_array (array): Array of simulated detections with length number
            of detections. A value of 1 indicates a detected clsuter and a
            value of 0 indicates an undetected cluster.

    Returns:
        percent_corr (float): Value for fraction of detections correctly
        labeled compared to ground truth.
    """

    assert len(gt) == np.shape(data_array)[
        0], "Number of GT detections must equal number of simulated detections"
    for i in range(len(gt)):
        assert gt[i] == 1 or gt[i] == 0, "Items in GT detections must equal 0 or 1"

    num_correct = 0

    for i in range(len(gt)):
        label = np.round(data_array[i][0])

        if gt[i] == 1 and label == 1:
            num_correct += 1
        elif gt[i] == 0 and label == 0:
            num_correct += 1

    percent_corr = num_correct / len(gt)

    return percent_corr


def is_in_image(x, y, a, L):
    """Determines if a square with defined vertices is contained in an image
    with larger dimensions

    Args:
        x (int): Value for the x coordinate of the top left corner of the
            square of interest
        y (int): Value for the y coordinate of the top left corner of the
            square of interest
        a (int): Value for the side length of the square of interest
        L (int): Value for the dimensions of the larger image

    Returns:
        bool: Whether the square is contained in image dimensions
    """

    return (x + a <= (L - 1)) and (y + a <= (L - 1)) and (x >= 0) and (y >= 0)


def is_overlapping(x_list, y_list, a_list, x, y, a):
    # check if a square with left corner at x,y,a
    # overlaps with other squares with corner coordinates and side length in the list
    """Determines if a square overlaps with a list of other squares.

    Returns boolean, ``True`` if square overlaps with any of squares in list,
    ``False`` if it doesn't overlap with any of squares in list

    Args:
        x_list (list): List of x coordinates for top left corners of squares
            to be compared with square of interest.
        y_list (list): List of y coordinates for top left corners of squares
            to be compared with square of interest.
        a_list (list): List of side lengths of squares to be compared with
            square of interest.
        x (int): Value for the x coordinate of the top left corner of the
            square of interest
        y (int): Value for the y coordinate of the top left corner of the
            square of interest
        a (int): Value for the side length of the square of interest

    Returns:
        bool: Whether the square overlaps with any of squares in list.
    """

    assert len(x_list) == len(y_list) == len(
        a_list), "There must be the same number of x and y coordinates and side lengths"

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    a_list = np.array(a_list)
    not_overlapping = ((x + a) < x_list) | ((x_list + a_list) < x) | \
        ((y + a) < y_list) | ((y_list + a_list) < y)
    return not all(not_overlapping)


def add_gaussian_noise(image, m, s):
    """Adds gaussian random noise with mean m and standard deviation s
    to the input image.

    Args:
        image (numpy.array): 2D image to add noise.
        m: mean of gaussian random noise to be added to each pixel of image.
        s: standard deviation of gaussian random noise to be added to each
            pixel of image.

    Returns:
        numpy.array: The noisy image.
    """
    row, col = image.shape
    gauss = np.random.normal(m, s, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def gaussian_spot_image_generator(L,
                                  N_min,
                                  N_max,
                                  sigma_mean,
                                  sigma_std,
                                  A_mean=1,
                                  A_std=0,
                                  noise_mean=0,
                                  noise_std=0,
                                  segmask=False,
                                  yield_pos=False):
    """Generates random images of Gaussian spots with random uniformly
    distributed center positions in the image area, i.e. in ``[0,L-1]*[0,L-1].``
    The number of spots in an image is uniformly distributed in `[N_min, N_max]`.
    Each spot is a gaussian with standard deviation normally distributed with
    `sigma_mean`, `sigma_std`, and cutoff value of 0.5 (it is redrawn if a
    smaller value is drawn). The intensity of each spot is normally distributed.

    Args:
        L : generated image side length - the generated images have shape `(L,L)`
        N_min, N_max: the number of spots plotted in each image is uniformly
            distributed in `[N_min, N_max]`.
        sigma_mean, sigma_std: the mean and standard deviation of the normally
            distributed spot width sigma (i.e. each spot is a Gaussian with
            standard deviation sigma).
        A_mean, A_std: the intensity of each spot is normally distributed in
            with mean `A_mean`, and standard deviation `A_std`.
        yield_pos: if ``True``, will yield lists of x and y positions and bounding
            boxes in addition to image and label image.
        noise_mean, noise_std: mean and std of white noise to be added to
            every pixel of the image

    Returns:
        img: `(L, L)` numpy array simulated image
        label: `(L, L)` numpy array of - 0 background, 1 for pixel of (rounded)
        spot center if segmask is ``False`` segmentation mask if `segmask` is
        ``True`` (pixel values are `0` in background, `1,...,N` for pixels belonging
        to the `N` spots in the image)
    """

    while True:  # keep yielding images forever
        img = np.zeros((L, L))     # create the image

        # coordinates for grid of pixels
        X = np.arange(0, L, 1)
        Y = np.arange(0, L, 1)
        X, Y = np.meshgrid(X, Y)

        # draw the number of dots that will be created in the image - an integer N_min <= N <= N_max
        N = random.randint(N_min, N_max)
        # allocate variables to store dot positions
        x_list = []
        y_list = []
        sigma_list = []
        bboxes = []
        # loop on all dots, generate the intensity for each dot and sum into img
        for ind in range(N):
            # draw the position (x,y) uniformly from [-0.5,L-0.5]*[-0.5,L-0.5]
            x = random.uniform(-0.5, L - 0.5)
            y = random.uniform(-0.5, L - 0.5)
            x_list.append(x)
            y_list.append(y)

            # draw the width of the gaussian
            sigma = random.gauss(sigma_mean, sigma_std)
            # while sigma < 0.5: # if sigma is too small, resample because it causes divergence
            #    sigma = random.gauss(sigma_mean, sigma_std)

            sigma_list.append(sigma)
            # draw the intensity from a normal distribution
            A = random.gauss(A_mean, A_std)

            # add a the bounding box inscribing the circle of radius sigma and center in (x,y) to
            # the list
            x1 = x - sigma
            x2 = x + sigma
            y1 = y - sigma
            y2 = y + sigma
            bboxes.append([x1, y1, x2, y2])

            # plot a gaussian
            # Mean vector and covariance matrix
            Z = A * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
            img += Z

            # add white noise to the image
            img = add_gaussian_noise(img, noise_mean, noise_std)
            # when white noise was added to 0 background, negative pixel values are obtained - set
            # them to zero
            img[img < 0] = 0

        # create label
        if segmask:
            # create segmentation mask
            label = np.zeros((L, L))     # create the image
            for spot_ind in range(len(x_list)):
                rr, cc = skimage.draw.disk(
                    (x_list[spot_ind], y_list[spot_ind]), sigma_list[spot_ind], shape=label.shape)
                label[cc, rr] = spot_ind + 1
        else:
            # create image with background 0, spot centers labeled with 1
            # use floor function since pixel with indices (i,j) covers the area [i,i+1] x [j,j+1]
            # and thus contains all coordinates of the form (i.x,j.x)
            x_ind = np.floor(x_list)
            y_ind = np.floor(y_list)
            label = np.zeros((L, L))     # create the image
            # pixels that are at the (rounded) center of a dot, are marked with 1
            label[y_ind.astype(int), x_ind.astype(int)] = 1

        if not yield_pos:
            yield (img, label)
        else:
            bboxes = np.array(bboxes)
            # reshape bboxes in case it is empty.
            bboxes = np.reshape(bboxes, (bboxes.shape[0], 4))
            yield (img, label, x_list, y_list, bboxes)
