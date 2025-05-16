# Copyright 2019-2024 The Van Valen Lab at the California Institute of
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

"""Image normalization methods"""

import logging

import numpy as np
import cv2 as cv


def mean_std_normalize(image, epsilon=1e-07):
    """Normalize image data by subtracting standard deviation pixel value
    and dividing by mean pixel value.

    Args:
        image (numpy.array): 4D numpy array of image data.
        epsilon (float): fuzz factor used in numeric expressions.

    Returns:
        numpy.array: normalized image data.
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    if not len(np.shape(image)) == 4:
        raise ValueError('Image must be 4D, input image shape was'
                         ' {}.'.format(np.shape(image)))

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = (img - img.mean()) / (img.std() + epsilon)
            image[batch, ..., channel] = normal_image
    return image


def min_max_normalize(image, clip=False):
    """Normalize image data by subtracting minimum pixel value and
     dividing by the maximum pixel value.

    Args:
        image (numpy.array): 4D numpy array of image data.
        clip (boolean): Defaults to false. Determines if pixel
            values are clipped by percentile.

    Returns:
        numpy.array: normalized image data.
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    if not len(np.shape(image)) == 4:
        raise ValueError('Image must be 4D, input image shape was'
                         ' {}.'.format(np.shape(image)))

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]

            if clip:
                img = np.clip(img, a_min=np.percentile(img, 0.01), a_max=np.percentile(img, 99.9))

            min_val = np.min(img)
            max_val = np.max(img)
            normal_image = (img - min_val) / (max_val - min_val)

            image[batch, ..., channel] = normal_image
    return image


def blurred_laplacian_of_gaussian(images, clip=False):
    """
    Applies a combination of bilateral filtering, Gaussian blurring, and Laplacian of Gaussian. 
    First denoises the images using a bilateral filter, then applies a Gaussian blur, 
    and finally enhances spots using the Laplacian of Gaussian method. 
    
    Currently, this function only supports single-channel images.
    
    Args:
        image (numpy.array): 4D numpy array of image data.
        clip (boolean): Defaults to false. Determines if pixel
            values are clipped by percentile.

    Returns:
        numpy.array: preprocessed image data.
    """
    # Combination: Bilateral denoising to preserve edges, then Laplacian of Gaussian for spot enhancement
    processed_images_list = []
    
    if not np.issubdtype(images.dtype, np.floating):
        logging.info('Converting image dtype to float')

    if not len(np.shape(images)) == 4:
        raise ValueError('Image must be 4D, input image shape was'
                         ' {}.'.format(np.shape(images)))
        
    for img_array in images:
        img = np.copy(img_array)
        
        if clip:
            img = np.clip(img, a_min=np.percentile(img, 0.01), a_max=np.percentile(img, 99.9))
                
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        img_float32 = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX).astype(np.float32)
        # Bilateral filter to denoise while maintaining spot edges
        bilateral = cv.bilateralFilter(img_float32, d=5, sigmaColor=0.09, sigmaSpace=9)
        # Slight Gaussian blur before Laplacian
        gauss = cv.GaussianBlur(bilateral, (3,3), 0)
        lap = cv.Laplacian(gauss, cv.CV_32F, ksize=3)
        abs_lap = np.abs(lap)
        lap_norm = cv.normalize(abs_lap, None, 0, 1, cv.NORM_MINMAX).astype(np.float32)
        if img_array.ndim == 3 and img_array.shape[2] == 1:
            lap_norm = lap_norm[:, :, np.newaxis]
        processed_images_list.append(lap_norm)
    return np.array(processed_images_list, dtype=np.float32)
