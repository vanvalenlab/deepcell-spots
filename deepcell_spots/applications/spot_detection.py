# Copyright 2016-2021 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
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
"""Nuclear segmentation application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import tensorflow as tf

from deepcell_spots.preprocessing_utils import min_max_normalize
from deepcell_spots.postprocessing_utils import y_annotations_to_point_list_max

from deepcell_spots.applications.application import Application
from deepcell_spots.losses import DotNetLosses
from deepcell_spots.dotnet import *

# MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
#               'saved-models/NuclearSegmentation-3.tar.gz')


class SpotDetection(Application):
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
        'name': 'general_nuclear_train_large', #update
        'other': 'Pooled nuclear data from HEK293, HeLa-S3, NIH-3T3, and RAW264.7 cells.' #update
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 1,
        'lr': 0.01,
        'lr_decay': 0.99,
        'training_seed': 0,
        'n_epochs': 10,
        'training_steps_per_epoch': 552,
        'validation_steps_per_epoch': 61 #not sure, just divided by 9
    }

    def __init__(self, model=None):

        if model is None:
            # archive_path = tf.keras.utils.get_file(
            #     'NuclearSegmentation.tgz', MODEL_PATH,
            #     file_hash='7fff56a59f453252f24967cfe1813abd',
            #     extract=True, cache_subdir='models'
            # )
            # model_path = os.path.splitext(archive_path)[0]
            model_path = '/data/20210331-training_data/models/em_model'
            model = tf.keras.models.load_model(model_path, custom_objects={'regression_loss':DotNetLosses.regression_loss,
                                                                             'classification_loss':DotNetLosses.classification_loss})

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
                postprocess_kwargs=None):
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
        if preprocess_kwargs is None:
            preprocess_kwargs = {}

        if postprocess_kwargs is None:
            postprocess_kwargs = {
                'threshold':0.9,
                'min_distance':1}

        return self._predict_segmentation(
            image,
            batch_size=batch_size,
            image_mpp=image_mpp,
            pad_mode=pad_mode,
            preprocess_kwargs=preprocess_kwargs,
            postprocess_kwargs=postprocess_kwargs)
