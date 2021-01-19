# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
"""Feature pyramid network utility functions"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, DepthwiseConv2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.keras.layers import BatchNormalization

from deepcell.layers import UpsampleLike
from deepcell.utils.misc_utils import get_sorted_keys


def __create_semantic_head(pyramid_dict,
                           input_target=None,
                           n_classes=3,
                           n_filters=128,
                           n_dense=128,
                           semantic_id=0,
                           ndim=2,
                           include_top=False,
                           target_level=2,
                           upsample_type='upsamplelike',
                           interpolation='bilinear',
                           **kwargs):
    """Creates a semantic head from a feature pyramid network.
    Args:
        pyramid_dict (dict): Dictionary of pyramid names and features.
        input_target (tensor): Optional tensor with the input image.
        n_classes (int): The number of classes to be predicted.
        n_filters (int): The number of convolutional filters.
        n_dense (int): Number of dense filters.
        semantic_id (int): ID of the semantic head.
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        include_top (bool): Whether to include the final layer of the model
        target_level (int): The level we need to reach. Performs
            2x upsampling until we're at the target level.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.
    Raises:
        ValueError: ``ndim`` must be 2 or 3
        ValueError: ``interpolation`` not in ``['bilinear', 'nearest']``
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``
    Returns:
        tensorflow.keras.Layer: The semantic segmentation head
    """
    # Check input to ndims
    if ndim not in {2, 3}:
        raise ValueError('ndim must be either 2 or 3. '
                         'Received ndim = {}'.format(ndim))

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode "{}" not supported. '
                         'Choose from {}.'.format(
                             interpolation, list(acceptable_interpolation)))

    # Check input to upsample_type
    acceptable_upsample = {'upsamplelike', 'upsampling2d', 'upsampling3d'}
    if upsample_type not in acceptable_upsample:
        raise ValueError('Upsample method "{}" not supported. '
                         'Choose from {}.'.format(
                             upsample_type, list(acceptable_upsample)))

    # Check that there is an input_target if upsamplelike is used
    if upsample_type == 'upsamplelike' and input_target is None:
        raise ValueError('upsamplelike requires an input_target.')

    conv = Conv2D if ndim == 2 else Conv3D
    conv_kernel = (1,) * ndim

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if n_classes == 1:
        include_top = False

    # Get pyramid names and features into list form
    pyramid_names = get_sorted_keys(pyramid_dict)
    pyramid_features = [pyramid_dict[name] for name in pyramid_names]

    # Reverse pyramid names and features
    pyramid_names.reverse()
    pyramid_features.reverse()

    # Previous method of building feature pyramids
    # semantic_features, semantic_names = [], []
    # for N, P in zip(pyramid_names, pyramid_features):
    #     # Get level and determine how much to upsample
    #     level = int(re.findall(r'\d+', N)[0])
    #
    #     n_upsample = level - target_level
    #     target = semantic_features[-1] if len(semantic_features) > 0 else None
    #
    #     # Use semantic upsample to get semantic map
    #     semantic_features.append(semantic_upsample(
    #         P, n_upsample, n_filters=n_filters, target=target, ndim=ndim,
    #         upsample_type=upsample_type, interpolation=interpolation,
    #         semantic_id=semantic_id))
    #     semantic_names.append('Q{}'.format(level))

    # Add all the semantic features
    # semantic_sum = semantic_features[0]
    # for semantic_feature in semantic_features[1:]:
    #     semantic_sum = Add()([semantic_sum, semantic_feature])

    # TODO: bad name but using the same name more clearly indicates
    # how to integrate the previous version
    semantic_sum = pyramid_features[-1]

    # Final upsampling
    # min_level = int(re.findall(r'\d+', pyramid_names[-1])[0])
    # n_upsample = min_level - target_level
    n_upsample = target_level
    x = semantic_upsample(semantic_sum, n_upsample,
                          # n_filters=n_filters,  # TODO: uncomment and retrain
                          target=input_target, ndim=ndim,
                          upsample_type=upsample_type, semantic_id=semantic_id,
                          interpolation=interpolation)

    # Apply conv in place of previous tensor product
    x = conv(n_dense, conv_kernel, strides=1, padding='same',
             name='conv_0_semantic_{}'.format(semantic_id))(x)
    x = BatchNormalization(axis=channel_axis,
                           name='batch_normalization_0_semantic_{}'.format(semantic_id))(x)
    x = Activation('relu', name='relu_0_semantic_{}'.format(semantic_id))(x)

    # Apply conv and softmax layer
    x = conv(n_classes, conv_kernel, strides=1,
             padding='same', name='conv_1_semantic_{}'.format(semantic_id))(x)

    if include_top:
        x = Softmax(axis=channel_axis,
                    dtype=K.floatx(),
                    name='semantic_{}'.format(semantic_id))(x)
    else:
        x = Activation('relu',
                       dtype=K.floatx(),
                       name='semantic_{}'.format(semantic_id))(x)

    return x