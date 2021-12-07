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
"""Spot detection application"""

from __future__ import absolute_import, division, print_function

import glob
import os

import tensorflow as tf
from deepcell_spots.applications.spots_application import SpotsApplication
from deepcell_spots.dotnet import *
from deepcell_spots.dotnet_losses import DotNetLosses
from deepcell_spots.postprocessing_utils import y_annotations_to_point_list_max
from deepcell_spots.preprocessing_utils import min_max_normalize

MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
              'saved-models/SpotDetection-3.tar.gz')


class SpotDetection(SpotsApplication):
    """Loads a :mod:`deepcell.model_zoo.panopticnet.PanopticNet` model
    for nuclear segmentation with pretrained weights.
    The ``predict`` method handles prep and post processing steps
    to return a labeled image.
    Example:
    .. code-block:: python
        from skimage.io import imread
        from deepcell_spots.applications import SpotDetection
        # Load the image
        im = imread('HeLa_nuclear.png')
        # Expand image dimensions to rank 4
        im = np.expand_dims(im, axis=-1)
        im = np.expand_dims(im, axis=0)
        # Create the application
        app = SpotDetection()
        # create the lab
        labeled_image = app.predict(im)
    Args:
        model (tf.keras.Model): The model to load. If ``None``,
            a pre-trained model will be downloaded.
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': 'general_train',  # update
        'other': """Pooled FISH data including MERFISH data
                    and SunTag viral RNA data"""  # update
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
            # model_path = '/data/20210331-training_data/models/em_model'
            archive_path = tf.keras.utils.get_file(
                'SpotDetection.tgz', MODEL_PATH,
                file_hash='2b9a46087b25e9aab20a2c9f67f4f559',
                extract=True, cache_subdir='models'
            )
            model_path = os.path.splitext(archive_path)[0]
            model = tf.keras.models.load_model(
                model_path, custom_objects={
                    'regression_loss': DotNetLosses.regression_loss,
                    'classification_loss': DotNetLosses.classification_loss
                }
            )

        super(SpotDetection, self).__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=0.65,
            preprocessing_fn=min_max_normalize,
            postprocessing_fn=y_annotations_to_point_list_max,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)

    def predict(self,
                image,
                batch_size=4,
                image_mpp=None,
                pad_mode='reflect',
                preprocess_kwargs=None,
                postprocess_kwargs=None,
                threshold=0.9):
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
            threshold (float): Probability threshold for a pixel to be
                                considered as a spot.
        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than
                expected by the model.
            ValueError: Input data must match required number of channels.
        Returns:
            numpy.array: Coordinate locations of detected spots.
        """

        if threshold < 0 or threshold > 1:
            raise ValueError("""Enter a probability threshold value between
                                0 and 1.""")

        if preprocess_kwargs is None:
            preprocess_kwargs = {}

        if postprocess_kwargs is None:
            postprocess_kwargs = {
                'threshold': threshold,
                'min_distance': 1}

        return self._predict_segmentation(
            image,
            batch_size=batch_size,
            image_mpp=image_mpp,
            pad_mode=pad_mode,
            preprocess_kwargs=preprocess_kwargs,
            postprocess_kwargs=postprocess_kwargs)
