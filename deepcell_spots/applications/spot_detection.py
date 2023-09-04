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

"""Spot detection application"""

from __future__ import absolute_import, division, print_function

import os
import timeit

import tensorflow as tf
from pathlib import Path
from deepcell.applications import Application
from deepcell.utils import fetch_data, extract_archive

from deepcell_spots.dotnet_losses import DotNetLosses
from deepcell_spots.utils.postprocessing_utils import y_annotations_to_point_list_max
from deepcell_spots.utils.preprocessing_utils import min_max_normalize


MODEL_KEY = 'models/SpotDetection-7.tar.gz'
MODEL_NAME = 'SpotDetection'
MODEL_HASH = 'f52d473ad7e4ce33472f1a9a9cae2d85'


def output_to_dictionary(output_images, output_names):
    """Formats model output from list to dictionary.

    Args:
        output_images (list): Model output list of length 2 containing
            classification prediction and regression prediction
        output_names (list): Model output names

    Returns:
        Dictionary with output names as keys and output images as values
    """
    return {name: pred for name, pred in zip(output_names,
                                             output_images)}


class SpotDetection(Application):
    """Loads a :mod:`deepcell.model_zoo.featurenet.FeatureNet` model
    for fluorescent spot detection with pretrained weights.

    The ``predict`` method handles prep and post processing steps
    to return a list of spot locations.

    Example::

        from skimage.io import imread
        from deepcell_spots.applications import SpotDetection

        # Load the image
        im = imread('spots_image.png')

        # Expand image dimensions to rank 4
        im = np.expand_dims(im, axis=-1)
        im = np.expand_dims(im, axis=0)

        # Create the application
        app = SpotDetection()

        # Find spot locations
        coords = app.predict(im)

    Args:
        model (tf.keras.Model): The model to load. If `None`,
            a pre-trained model will be downloaded.
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': 'general_train',
        'other': """Pooled FISH data including MERFISH data
                    and SunTag viral RNA data"""
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 1,
        'lr': 0.01,
        'lr_decay': 0.99,
        'training_seed': 0,
        'n_epochs': 10,
        'training_steps_per_epoch': 552
    }

    def __init__(self, model=None):

        if model is None:
            cache_subdir = "models"
            model_dir = Path.home() / ".deepcell" / "models"

            archive_path = fetch_data(
                asset_key=MODEL_KEY,
                cache_subdir=cache_subdir,
                file_hash=MODEL_HASH
            )
            extract_archive(archive_path, model_dir)
            model_path = model_dir / MODEL_NAME
            model = tf.keras.models.load_model(
                model_path, custom_objects={
                    'regression_loss': DotNetLosses.regression_loss,
                    'classification_loss': DotNetLosses.classification_loss
                }
            )

        super(SpotDetection, self).__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=0.1,
            preprocessing_fn=min_max_normalize,
            postprocessing_fn=y_annotations_to_point_list_max,
            format_model_output_fn=output_to_dictionary,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)

    def _format_model_output(self, output_images):
        """Applies formatting function the output from the model if one was
        provided. Otherwise, returns the unmodified model output.

        Args:
            output_images: stack of untiled images to be reformatted

        Returns:
            dict or list: reformatted images stored as a dict, or input
            images stored as list if no formatting function is specified.
        """
        if self.format_model_output_fn is not None:
            formatted_images = self.format_model_output_fn(output_images, self.model.output_names)
            return formatted_images
        else:
            return output_images

    def _postprocess(self, image, **kwargs):
        """Applies postprocessing function to image if one has been defined.
        Differs from parent class in that it returns a set of coordinate spot
        locations, so handling of dimensions differs.

        Otherwise returns unmodified image.
        Args:
            image (numpy.array or list): Input to postprocessing function
                either an ``numpy.array`` or list of ``numpy.arrays``.
        Returns:
            list: coordinate spot locations
        """
        if self.postprocessing_fn is not None:
            t = timeit.default_timer()
            self.logger.debug('Post-processing results with %s and kwargs: %s',
                              self.postprocessing_fn.__name__, kwargs)

            output = self.postprocessing_fn(image, **kwargs)

            self.logger.debug('Post-processed results with %s in %s s',
                              self.postprocessing_fn.__name__,
                              timeit.default_timer() - t)

        elif isinstance(image, list) and len(image) == 1:
            output = image[0]
        else:
            output = image

        return output

    def _predict(self,
                 image,
                 batch_size=4,
                 pad_mode='constant',
                 preprocess_kwargs={},
                 postprocess_kwargs={}):
        """Generates a list of coordinate spot locations of the input running
        prediction with appropriate pre and post processing functions.
        This differs from parent Application class which returns a labeled image.

        Input images are required to have 4 dimensions
        `[batch, x, y, channel]`.

        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                `[batch, x, y, channel]`.
            batch_size (int): Number of images to predict on per batch.
            pad_mode (str): The padding mode, one of `"constant"` or `"reflect"`.
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

        # Generate model outputs
        output_images = self._run_model(
            image=image, batch_size=batch_size,
            pad_mode=pad_mode, preprocess_kwargs=preprocess_kwargs
        )

        # Postprocess predictions to create label image
        predicted_spots = self._postprocess(output_images, **postprocess_kwargs)

        return predicted_spots

    def predict(self,
                image,
                batch_size=4,
                pad_mode='reflect',
                preprocess_kwargs=None,
                postprocess_kwargs=None,
                threshold=0.95,
                clip=True):
        """Generates a list of coordinate spot locations of the input
        running prediction with appropriate pre and post processing
        functions.

        Input images are required to have 4 dimensions
        `[batch, x, y, channel]`.

        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                `[batch, x, y, channel]`.
            batch_size (int): Number of images to predict on per batch.
            pad_mode (str): The padding mode, one of `"constant"` or `"reflect"`.
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.
            threshold (float): Probability threshold for a pixel to be
                considered as a spot.
            clip (bool): Determines if pixel values will be clipped by percentile.
                Defaults to `True`.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than
                expected by the model.
            ValueError: Input data must match required number of channels.
            ValueError: Threshold value must be between 0 and 1.

        Returns:
            numpy.array: Coordinate locations of detected spots.
        """

        if threshold < 0 or threshold > 1:
            raise ValueError('Threshold value must be between 0 and 1.')

        if preprocess_kwargs is None:
            preprocess_kwargs = {
                'clip': clip}

        if postprocess_kwargs is None:
            postprocess_kwargs = {
                'threshold': threshold,
                'min_distance': 1}

        return self._predict(
            image,
            batch_size=batch_size,
            pad_mode=pad_mode,
            preprocess_kwargs=preprocess_kwargs,
            postprocess_kwargs=postprocess_kwargs)
