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

import numpy as np
import tensorflow as tf

from deepcell.applications import CytoplasmSegmentation
from deepcell_spots.applications import SpotDetection
from deepcell_spots.singleplex import match_spots_to_cells
from deepcell_toolbox.processing import histogram_normalization
from deepcell_toolbox.deep_watershed import deep_watershed

class Polaris(object):
    #TODO fill out example
    """Loads a :mod:`deepcell.model_zoo.featurenet.FeatureNet` model
    for fluorescent spot detection with pretrained weights and a
    :mod:`deepcell.model_zoo.panopticnet.PanopticNet` model for
    cytoplasm segmentation with pretrained weights.
    The ``predict`` method handles prep and post processing steps
    to return a labeled image.
    Example:
    Args:
        model (tf.keras.Model): The model to load. If ``None``,
            a pre-trained model will be downloaded.
    """

    def __init__(self):

        model = tf.keras.models.load_model('../notebooks/models/CytoplasmSegmentation')

        self.spots_app = SpotDetection()
        self.segmentation_app = CytoplasmSegmentation(model=model)

        self.segmentation_app.preprocessing_fn = histogram_normalization
        self.segmentation_app.postprocessing_fn = deep_watershed

    def predict(self,
                image,
                image_mpp=None,
                # segmentation
                cytoplasm_channel=0,
                # spots
                spots_channel=1,
                threshold=0.95,
                clip=False):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``.

        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.
            pad_mode (str): The padding mode, one of "constant" or "reflect".
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Labeled image
        """

        if threshold < 0 or threshold > 1:
            raise ValueError("""Enter a probability threshold value between
                                0 and 1.""")

        cytoplasm_image = image[:, :, :, cytoplasm_channel]
        cytoplasm_image = np.expand_dims(cytoplasm_image, axis=-1)

        spots_image = image[:, :, :, spots_channel]
        spots_image = np.expand_dims(spots_image, axis=-1)

        segmentation_result = self.segmentation_app.predict(cytoplasm_image,
                                                       image_mpp=image_mpp)
        spots_result = self.spots_app.predict(spots_image,
                                         threshold=threshold,
                                         clip=clip)

        result = []
        for i in range(len(spots_result)):
            spots_dict = match_spots_to_cells(segmentation_result[i:i+1, :, :, :],
                                              spots_result[i])

            result.append({'spots_assignment': spots_dict,
                           'cell_segmentation': segmentation_result[i:i+1, :, :, :],
                           'spot_locations': spots_result[i]})

        return result
