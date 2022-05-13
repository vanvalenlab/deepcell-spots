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

"""CNN architechture with classification and regression outputs for dot center detection"""

from deepcell.layers import TensorProduct
from deepcell.model_zoo import bn_feature_net_skip_2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Activation, BatchNormalization,
                                            Conv2D, Input, Lambda, Permute,
                                            Reshape, Softmax)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


def default_heads(input_shape, num_classes):
    """
    Create a list of the default heads for dot detection center pixel detection
    and offset regression

    Args:
        input_shape
        num_classes

    Returns:
        list(tuple): A list of tuple, where the first element is the name of
            the submodel and the second element is the submodel itself.
    """
    return [
        ('offset_regression', offset_regression_head(
            input_shape=input_shape)),
        ('classification', classification_head(
            input_shape, n_features=num_classes))
    ]


def classification_head(input_shape,
                        n_features=2,
                        n_dense_filters=128,
                        reg=1e-5,
                        init='he_normal',
                        name='classification_head'):

    """Creates a classification head.

    Args:
        n_features (int): Number of output features (number of possible classes
            for each pixel).
        default is 2: contains point / does not contain point)
        reg (int): regularization value
        init (str): Method for initalizing weights.

    Returns:
        tensorflow.keras.Model for classification (softmax output)
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = []  # Create layers list (x) to store all of the layers.
    inputs = Input(shape=input_shape)
    x.append(inputs)
    x.append(TensorProduct(n_dense_filters, kernel_initializer=init,
                           kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))
    x.append(TensorProduct(n_features, kernel_initializer=init,
                           kernel_regularizer=l2(reg))(x[-1]))
    outputs = Softmax(axis=channel_axis)(x[-1])

    return Model(inputs=inputs, outputs=outputs, name=name)


def offset_regression_head(input_shape,
                           regression_feature_size=256,
                           name='offset_regression_head'):

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = Input(shape=input_shape)
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='offset_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = Conv2D(filters=2, name='offset_regression', **options)(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


def __build_model_heads(name, model, backbone_output):
    identity = Lambda(lambda x: x, name=name)
    return identity(model(backbone_output))


def dot_net_2D(receptive_field=13,
               input_shape=(256, 256, 1),
               inputs=None,
               n_skips=3,
               norm_method='std',
               padding_mode='reflect',
               **kwargs):

    inputs = Input(shape=input_shape)

    featurenet_model = bn_feature_net_skip_2D(
        receptive_field=receptive_field,
        input_shape=inputs.get_shape().as_list()[1:],
        inputs=inputs,
        n_features=2,  # segmentation mask (is_background, is_dot)
        norm_method=norm_method,
        padding_mode=padding_mode,
        fgbg_model=None,
        n_conv_filters=32,
        n_dense_filters=128,
        n_skips=n_skips,
        last_only=True,
        include_top=False)

    featurenet_output = featurenet_model(inputs)

    # add 2 heads: 1 for center pixel classification
    # (should be 1 for pixel which has center, 0 otherwise),
    # and 1 for center location regression
    # (size of classification output where pixel value = signed x/y distance to nearest max
    # of classification)
    # softmax top (as in include_top==True for bn_feature_net_2D):

    input_shape = featurenet_output.get_shape().as_list()[1:]

    # 2 classes: contains / does not contain dot center
    head_submodels = default_heads(input_shape=input_shape, num_classes=2)
    dot_head = [__build_model_heads(n, m, featurenet_output)
                for n, m in head_submodels]

    model = Model(inputs=inputs, outputs=dot_head)
    return model
