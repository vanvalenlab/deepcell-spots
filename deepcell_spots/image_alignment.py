# Copyright 2019-2021 The Van Valen Lab at the California Institute of
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

import os

import cv2
import numpy as np


def read_images(root_dir, dataorg, verbose=True):
    """Reads in image files from given directories and parses them into dictionaries of different
    types.

    Args:
        root_dir (str): Directory containing all image files
        image_files (list): List of image names (str) in root directory. Paths must be to images
            must be saved in .npy format.
        dataorg (pandas.DataFrame): Data frame with required columns 'fileName' (item in
            image_files), 'readoutName' (unique ID name given to each channel in each image),
            'fiducialFrame' (frame number for image to be used for alignment), 'cytoplasmFrame'
            (frame number for image to be used for cell segmentation)
        verbose (bool, optional): Boolean determining if file names are printed as they are
            processed. Defaults to True.

    Returns:
        max_im_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            maximum intensity projections of frames associated with that readout name
        fiducial_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            fiducial channel (image used for alignment) for each readout name
            (multiple readout names may have the same)
        cytoplasm_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            cytoplasm label image for each readout name (multiple readout names may have the same)
    """

    max_im_dict = {}
    reference_dict = {}
    cytoplasm_dict = {}

    # Grab all file names
    image_files = list(dataorg['fileName'].unique())

    # Remove invalid file names
    image_files = [x for x in image_files if type(x) == str]

    for i in range(len(image_files)):

        # Slice data organization table to get information for this image stack
        round_df = dataorg.loc[dataorg['fileName'] == image_files[i]]
        image_file = os.path.join(root_dir, image_files[i])

        # Load image stack
        image_stack = np.load(image_file)

        # Grab ID names for each image in stack
        rounds = round_df['readoutName'].values
        rounds = [item for item in rounds if 'Spots' in item]

        for item in rounds:
            if verbose:
                print('Working on: {}'.format(item))

            # Get frames associated with a round in the image stack
            frame_list = round_df['frame'].loc[round_df['readoutName'] == item].values[0]
            frame_list = frame_list.strip('][').split(', ')
            frame_list = np.array(frame_list).astype(int)

            start_frame = frame_list[0]
            end_frame = frame_list[-1]

            # Maximum projection
            max_im = np.max(image_stack[:, :, start_frame:end_frame + 1], axis=2)

            # Clip outlier high pixel values
            im = np.clip(max_im, np.min(max_im), np.percentile(max_im, 99.9))
            im = np.expand_dims(im, axis=[0, -1])

            max_im_dict[item] = im

            ref_frame = dataorg.loc[dataorg['readoutName'] == 'Reference']['frame'].values[0]
            ref_frame = ref_frame.strip('][').split(', ')
            ref_frame = np.array(ref_frame).astype(int)
            ref_frame = np.mean([ref_frame[0], ref_frame[-1]]).astype(int)
            reference_dict[item] = np.expand_dims(image_stack[:, :, ref_frame], axis=[0, -1])

            cyto_frame = dataorg.loc[dataorg['readoutName'] == 'Cytoplasm']['frame'].values[0]
            cyto_frame = cyto_frame.strip('][').split(', ')
            cyto_frame = np.array(cyto_frame).astype(int)
            cyto_frame = np.mean([cyto_frame[0], cyto_frame[-1]]).astype(int)
            cytoplasm_dict[item] = np.expand_dims(image_stack[:, :, cyto_frame], axis=[0, -1])

    return(max_im_dict, reference_dict, cytoplasm_dict)


def align_images(image_dict, reference_dict):
    """Aligns input images with alignment transformation learned from reference images.

    Args:
        image_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            images to be aligned for each readout name
        reference_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            fiducial channel (image used for alignment) for each readout name
            (multiple readout names may have the same reference image)

    Returns:
        aligned_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            images from image_dict that have been aligned by transformations learned from
            images from reference_dict
    """

    aligned_dict = {}

    image_keys = list(image_dict.keys())
    num_images = len(image_keys)

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    orb = cv2.ORB_create(MAX_FEATURES)
    reference_im = cv2.convertScaleAbs(reference_dict[image_keys[0]][0, :, :, :],
                                       alpha=(255.0 / 65535.0))
    keypoints2, descriptors2 = orb.detectAndCompute(reference_im, None)

    for idx in range(num_images):
        im1 = cv2.convertScaleAbs(reference_dict[image_keys[idx]][0, :, :, :],
                                  alpha=(255.0 / 65535.0))
        orb = cv2.ORB_create(MAX_FEATURES)

        keypoints1, descriptors1 = orb.detectAndCompute(im1, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(reference_im, keypoints2, im1, keypoints1, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width = reference_im.shape
        target_im = image_dict[image_keys[idx]][0, :, :, :]
        aligned_im = cv2.warpPerspective(target_im, h, (width, height))

        aligned_dict[image_keys[idx]] = np.expand_dims(aligned_im, axis=[0, -1])

    return aligned_dict


def crop_images(aligned_dict):
    """Crops images to remove zero-value pixels resulting from image alignment.

    Args:
        aligned_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            images from image_dict that have been aligned with align_images
    Returns:
        crop_dict (dict): Dictionary where keys are image IDs ('readoutName') and values are
            images from image_dict that have been aligned with align_images with zero-value
            pixels cropped out
    """
    crop_dict = {}

    crop_bool = np.array(list(aligned_dict.values())) > 0
    crop_bool_all = np.min(crop_bool, axis=0)

    top = 0
    while np.array([crop_bool_all[0, :, :, 0][top] == 0]).all():
        top += 1
    bottom = np.shape(crop_bool_all)[1] - 1
    while np.array([crop_bool_all[0, :, :, 0][bottom] == 0]).all():
        bottom -= 1

    left = 0
    while np.array([crop_bool_all[0, :, :, 0][:, left] == 0]).all():
        left += 1
    right = np.shape(crop_bool_all)[2] - 1
    while np.array([crop_bool_all[0, :, :, 0][:, right] == 0]).all():
        right -= 1

    for item in aligned_dict.keys():
        # increment one more because sometimes low value pixels at edges of image from alignment
        crop_dict[item] = aligned_dict[item][:, top + 1:bottom, left + 1:right, :]

    return(crop_dict)
