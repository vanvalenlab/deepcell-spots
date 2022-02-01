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
"""Singleplex FISH analysis application"""

from __future__ import absolute_import, division, print_function

import os
import timeit
import warnings

import numpy as np
import tensorflow as tf

from deepcell.applications import CytoplasmSegmentation
from deepcell.applications import NuclearSegmentation
from deepcell_spots.applications import SpotDetection
from deepcell_spots.singleplex import match_spots_to_cells
from deepcell_toolbox.processing import histogram_normalization
from deepcell_toolbox.deep_watershed import deep_watershed
from tensorflow.python.platform.tf_logging import warning


class Polaris(object):
    """Loads spot detection and cell segmentation applications
    from deepcell_spots and deepcell_tf, respectively.
    The ``predict`` method calls the predict method of each
    application.
    Example:
    .. code-block:: python
        from skimage.io import imread
        from deepcell_spots.applications import Polaris
        # Load the images
        spots_im = imread('spots_image.png')
        cyto_im = imread('cyto_image.png')
        # Expand image dimensions to rank 4
        spots_im = np.expand_dims(spots_im, axis=[0,-1])
        cyto_im = np.expand_dims(cyto_im, axis=[0,-1])
        # Create the application
        app = Polaris()
        # Find the spot locations
        result = app.predict(spots_image=spots_im,
                             segmentation_image=cyto_im)
        spots_dict = result[0]['spots_assignment']
        labeled_im = result[0]['cell_segmentation']
        coords = result[0]['spot_locations']
    Args:
        segmentation_model (tf.keras.Model): The model to load.
            If ``None``, a pre-trained model will be downloaded.
        segmentation_compartment (str): The cellular compartment
            for generating segmentation predictions. Valid values
            are 'cytoplasm', 'nucleus', 'no segmentation'. Defaults
            to 'cytoplasm'.
        spots_model (tf.keras.Model): The model to load.
            If ``None``, a pre-trained model will be downloaded.
    """

    def __init__(self,
                 segmentation_model=None,
                 segmentation_compartment='cytoplasm',
                 spots_model=None):

        self.spots_app = SpotDetection(model=spots_model)

        valid_compartments = ['cytoplasm', 'nucleus', 'no segmentation']
        if segmentation_compartment not in valid_compartments:
            raise ValueError('Invalid compartment supplied: {}. '
                             'Must be one of {}'.format(segmentation_compartment,
                                                        valid_compartments))

        if segmentation_compartment == 'cytoplasm':
            self.segmentation_app = CytoplasmSegmentation(model=segmentation_model)
            self.segmentation_app.preprocessing_fn = histogram_normalization
            self.segmentation_app.postprocessing_fn = deep_watershed
        elif segmentation_compartment == 'nucleus':
            self.segmentation_app = NuclearSegmentation(model=segmentation_model)
        else:
            self.segmentation_app = None
            warnings.warn('No segmentation application instantiated.')

    def predict(self,
                spots_image,
                segmentation_image=None,
                image_mpp=None,
                spots_threshold=0.95,
                spots_clip=False):
        """Generates prediction output consisting of a labeled cell segmentation image,
        detected spot locations, and a dictionary of spot locations assigned to labeled
        cells of the input.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``. Channel dimension should be 2.

        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            spots_image (numpy.array): Input image for spot detection with shape
                ``[batch, x, y, channel]``.
            segmentation_image (numpy.array): Input image for cell segmentation with shape
                ``[batch, x, y, channel]``. Defaults to None.
            image_mpp (float): Microns per pixel for ``image``.
            spots_threshold (float): Probability threshold for a pixel to be
                considered as a spot.
            spots_clip (bool): Determines if pixel values will be clipped by percentile.
                Defaults to false.
        Raises:
            ValueError: Threshold value must be between 0 and 1.
            ValueError: Segmentation application must be instantiated if segmentation
                image is defined.

        Returns:
            list: List of dictionaries, length equal to batch dimension.
        """

        if spots_threshold < 0 or spots_threshold > 1:
            raise ValueError('Threshold of %s was input. Threshold value must be '
                             'between 0 and 1.'.format())

        spots_result = self.spots_app.predict(spots_image,
                                              threshold=spots_threshold,
                                              clip=spots_clip)

        if segmentation_image is not None:
            if not self.segmentation_app:
                raise ValueError('Segmentation application must be instantiated if '
                                 'segmentation image is defined.')
            else:
                segmentation_result = self.segmentation_app.predict(segmentation_image,
                                                                    image_mpp=image_mpp)
                result = []
                for i in range(len(spots_result)):
                    spots_dict = match_spots_to_cells(segmentation_result[i:i + 1],
                                                      spots_result[i])

                    result.append({'spots_assignment': spots_dict,
                                   'cell_segmentation': segmentation_result[i:i + 1],
                                   'spot_locations': spots_result[i]})

        else:
            result = spots_result

        return result
