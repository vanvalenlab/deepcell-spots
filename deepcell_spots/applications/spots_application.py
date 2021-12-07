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
"""Base class for applications"""

from __future__ import absolute_import, division, print_function

import logging
import timeit

import numpy as np
from deepcell.applications import Application
from deepcell_toolbox.utils import resize, tile_image, untile_image


class SpotsApplication(Application):
    """Application object that takes a model with weights
    and manages predictions
    Args:
        model (tensorflow.keras.Model): ``tf.keras.Model``
            with loaded weights.
        model_image_shape (tuple): Shape of input expected by ``model``.
        dataset_metadata (str or dict): Metadata for the data that
            ``model`` was trained on.
        model_metadata (str or dict): Training metadata for ``model``.
        model_mpp (float): Microns per pixel resolution of the
            training data used for ``model``.
        preprocessing_fn (function): Pre-processing function to apply
            to data prior to prediction.
        postprocessing_fn (function): Post-processing function to apply
            to data after prediction.
            Must accept an input of a list of arrays and then
            return a single array.
        format_model_output_fn (function): Convert model output
            from a list of matrices to a dictionary with keys for
            each semantic head.
    Raises:
        ValueError: ``preprocessing_fn`` must be a callable function
        ValueError: ``postprocessing_fn`` must be a callable function
        ValueError: ``model_output_fn`` must be a callable function
    """

    def _postprocess(self, image, **kwargs):
        """Applies postprocessing function to image if one has been defined.
        Differs from parent class in that it returns a set of coordinate spot
        locations, so handling of dimensions differs.

        Otherwise returns unmodified image.
        Args:
            image (numpy.array or list): Input to postprocessing function
                either an ``numpy.array`` or list of ``numpy.arrays``.
        Returns:
            numpy.array: labeled image
        """
        if self.postprocessing_fn is not None:
            t = timeit.default_timer()
            self.logger.debug('Post-processing results with %s and kwargs: %s',
                              self.postprocessing_fn.__name__, kwargs)

            image = self.postprocessing_fn(image, **kwargs)

            self.logger.debug('Post-processed results with %s in %s s',
                              self.postprocessing_fn.__name__,
                              timeit.default_timer() - t)

        elif isinstance(image, list) and len(image) == 1:
            image = image[0]

        return image

    def _run_model(self,
                   image,
                   batch_size=4,
                   pad_mode='constant',
                   preprocess_kwargs={}):
        """Run the model to generate output probabilities on the data.
        Args:
            image (numpy.array): Image with shape ``[batch, x, y, channel]``
            batch_size (int): Number of images to predict on per batch.
            pad_mode (str): The padding mode, one of "constant" or "reflect".
            preprocess_kwargs (dict): Keyword arguments to pass to
                the preprocessing function.
        Returns:
            numpy.array: Model outputs
        """
        # Preprocess image if function is defined
        image = self._preprocess(image, **preprocess_kwargs)

        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = self._tile_input(image, pad_mode=pad_mode)

        # Run images through model
        t = timeit.default_timer()
        output_tiles = self.model.predict(tiles, batch_size=batch_size)
        self.logger.debug('Model inference finished in %s s',
                          timeit.default_timer() - t)

        # Untile images
        output_images = self._untile_output(output_tiles, tiles_info)

        # restructure outputs into a dict if function provided
        formatted_images = self._format_model_output(output_images)

        return formatted_images

    def _predict_segmentation(self,
                              image,
                              batch_size=4,
                              image_mpp=None,
                              pad_mode='constant',
                              preprocess_kwargs={},
                              postprocess_kwargs={}):
        """Generates a list of coordinate spot locations of the input running
        prediction with appropriate pre and post processing functions.
        This differs from parent Application class which returns a labeled image.
        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``. Additional empty dimensions can be added
        using ``np.expand_dims``.
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
            ValueError: Input data must match required rank, calculated as one
                dimension more (batch dimension) than expected by the model.
            ValueError: Input data must match required number of channels.
        Returns:
            numpy.array: Coordinate spot locations
        """
        # Check input size of image
        if len(image.shape) != self.required_rank:
            raise ValueError('Input data must have {} dimensions. '
                             'Input data only has {} dimensions'.format(
                                 self.required_rank, len(image.shape)))

        if image.shape[-1] != self.required_channels:
            raise ValueError('Input data must have {} channels. '
                             'Input data only has {} channels'.format(
                                 self.required_channels, image.shape[-1]))

        # Resize image, returns unmodified if appropriate
        resized_image = self._resize_input(image, image_mpp)

        # Generate model outputs
        output_images = self._run_model(
            image=resized_image, batch_size=batch_size,
            pad_mode=pad_mode, preprocess_kwargs=preprocess_kwargs
        )

        # Resize output_images back to original resolution if necessary
        label_image = self._resize_output(output_images, image.shape)

        # Postprocess predictions to create label image
        predicted_spots = self._postprocess(label_image, **postprocess_kwargs)

        return predicted_spots
