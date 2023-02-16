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

from deepcell.layers import (TensorProduct, ImageNormalization2D,
                             ReflectionPadding2D, DilatedMaxPool2D,
                             Location2D)
from deepcell.model_zoo import bn_feature_net_skip_2D
from tensorflow.keras import backend as K
from tensorflow.keras import utils as keras_utils
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Conv2D, Input, Lambda, Permute,
                                     Reshape, Softmax, ZeroPadding2D,
                                     Concatenate, MaxPool2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization
import tensorflow_models as tfm


def bn_feature_net_2D_sngp(receptive_field=61,
                      input_shape=(256, 256, 1),
                      inputs=None,
                      n_features=3,
                      n_channels=1,
                      reg=1e-5,
                      n_conv_filters=64,
                      n_dense_filters=200,
                      VGG_mode=False,
                      init='he_normal',
                      norm_method='std',
                      location=False,
                      dilated=False,
                      padding=False,
                      padding_mode='reflect',
                      multires=False,
                      include_top=True):
    """Creates a 2D featurenet.
    Args:
        receptive_field (int): the receptive field of the neural network.
        input_shape (tuple): If no input tensor, create one with this shape.
        inputs (tensor): optional input tensor
        n_features (int): Number of output features
        n_channels (int): number of input channels
        reg (int): regularization value
        n_conv_filters (int): number of convolutional filters
        n_dense_filters (int): number of dense filters
        VGG_mode (bool): If ``multires``, uses ``VGG_mode``
            for multiresolution
        init (str): Method for initalizing weights.
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        location (bool): Whether to include a
            :mod:`deepcell.layers.location.Location2D` layer.
        dilated (bool): Whether to use dilated pooling.
        padding (bool): Whether to use padding.
        padding_mode (str): Type of padding, one of 'reflect' or 'zero'
        multires (bool): Enables multi-resolution mode
        include_top (bool): Whether to include the final layer of the model
    Returns:
        tensorflow.keras.Model: 2D FeatureNet
    """
    # Create layers list (x) to store all of the layers.
    # We need to use the functional API to enable the multiresolution mode
    x = []

    win = (receptive_field - 1) // 2

    if dilated:
        padding = True

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        row_axis = 2
        col_axis = 3

        if not dilated:
            input_shape = (n_channels, receptive_field, receptive_field)

    else:
        row_axis = 1
        col_axis = 2
        channel_axis = -1
        if not dilated:
            input_shape = (receptive_field, receptive_field, n_channels)

    if inputs is not None:
        if not K.is_keras_tensor(inputs):
            img_input = Input(tensor=inputs, shape=input_shape)
        else:
            img_input = inputs
        x.append(img_input)
    else:
        x.append(Input(shape=input_shape))

    x.append(ImageNormalization2D(norm_method=norm_method,
                                  filter_size=receptive_field)(x[-1]))

    if padding:
        if padding_mode == 'reflect':
            x.append(ReflectionPadding2D(padding=(win, win))(x[-1]))
        elif padding_mode == 'zero':
            x.append(ZeroPadding2D(padding=(win, win))(x[-1]))

    layers_to_concat = []

    rf_counter = receptive_field
    block_counter = 0
    d = 1

    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        x.append(SpectralNormalization(Conv2D(n_conv_filters, filter_size, dilation_rate=d,
                        kernel_initializer=init, padding='valid',
                        kernel_regularizer=l2(reg)))(x[-1]))
        x.append(BatchNormalization(axis=channel_axis)(x[-1]))
        x.append(Activation('relu')(x[-1]))

        block_counter += 1
        rf_counter -= filter_size - 1

        if block_counter % 2 == 0:
            if dilated:
                x.append(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(x[-1]))
                d *= 2
            else:
                x.append(MaxPool2D(pool_size=(2, 2))(x[-1]))

            if VGG_mode:
                n_conv_filters *= 2

            rf_counter = rf_counter // 2

            if multires:
                layers_to_concat.append(len(x) - 1)

    x.append(SpectralNormalization(Conv2D(n_dense_filters, (rf_counter, rf_counter), dilation_rate=d,
                    kernel_initializer=init, padding='valid',
                    kernel_regularizer=l2(reg)))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    if inputs is not None:
        real_inputs = keras_utils.get_source_inputs(x[0])
    else:
        real_inputs = x[0]

    model = Model(inputs=real_inputs, outputs=x[-1])

    return model


def bn_feature_net_skip_2D_sngp(receptive_field=61,
                           input_shape=(256, 256, 1),
                           inputs=None,
                           fgbg_model=None,
                           n_skips=2,
                           last_only=True,
                           norm_method='std',
                           padding_mode='reflect',
                           **kwargs):
    """Creates a 2D featurenet with skip-connections.
    Args:
        receptive_field (int): the receptive field of the neural network.
        input_shape (tuple): If no input tensor, create one with this shape.
        inputs (tensor): optional input tensor
        fgbg_model (tensorflow.keras.Model): Concatenate output of this model
            with the inputs as a skip-connection.
        last_only (bool): Model will only output the final prediction,
            and not return any of the underlying model predictions.
        n_skips (int): The number of skip-connections
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        padding_mode (str): Type of padding, one of 'reflect' or 'zero'
        kwargs (dict): Other model options defined in `~bn_feature_net_2D`
    Returns:
        tensorflow.keras.Model: 2D FeatureNet with skip-connections
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    inputs = Input(shape=input_shape)
    img = ImageNormalization2D(norm_method=norm_method,
                               filter_size=receptive_field)(inputs)

    models = []
    model_outputs = []

    if fgbg_model is not None:
        for layer in fgbg_model.layers:
            layer.trainable = False

        models.append(fgbg_model)
        fgbg_output = fgbg_model(inputs)
        if isinstance(fgbg_output, list):
            fgbg_output = fgbg_output[-1]
        model_outputs.append(fgbg_output)

    for _ in range(n_skips + 1):
        if model_outputs:
            model_input = Concatenate(axis=channel_axis)([img, model_outputs[-1]])
        else:
            model_input = img

        new_input_shape = model_input.get_shape().as_list()[1:]
        models.append(bn_feature_net_2D_sngp(receptive_field=receptive_field,
                                        input_shape=new_input_shape,
                                        norm_method=None,
                                        dilated=True,
                                        padding=True,
                                        padding_mode=padding_mode,
                                        **kwargs))
        model_outputs.append(models[-1](model_input))

    if last_only:
        model = Model(inputs=inputs, outputs=model_outputs[-1])
    elif fgbg_model is None:
        model = Model(inputs=inputs, outputs=model_outputs)
    else:
        model = Model(inputs=inputs, outputs=model_outputs[1:])

    return model


def default_heads(input_shape, num_classes):
    """
    Create a list of the default heads for spot detection.

    Args:
        input_shape (tuple): Shape of input image.
        num_classes (int): Number of output features (number of possible classes
            for each pixel).

    Returns:
        list(tuple): A list of tuples, where the first element is the name of
        the submodel and the second element is the submodel itself.
    """
    return [
        ('offset_regression', offset_regression_head(
            input_shape=input_shape)),
        ('classification', classification_head_sngp(
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
        input_shape (tuple): Shape of input image.
        n_features (int): Number of output features (number of possible classes
            for each pixel). Default is 2: contains point / does not contain
            point).
        n_dense_filters (int)
        reg (float): Regularization value
        init (str): Method for initalizing weights.

    Returns:
        tensorflow.keras.Model for classification (softmax output).
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


def classification_head_sngp(input_shape,
                        n_features=2,
                        n_dense_filters=128,
                        reg=1e-5,
                        init='he_normal',
                        name='classification_head'):
    """Creates a classification head.

    Args:
        input_shape (tuple): Shape of input image.
        n_features (int): Number of output features (number of possible classes
            for each pixel). Default is 2: contains point / does not contain
            point).
        n_dense_filters (int)
        reg (float): Regularization value
        init (str): Method for initalizing weights.

    Returns:
        tensorflow.keras.Model for classification (softmax output).
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = []  # Create layers list (x) to store all of the layers.
    inputs = Input(shape=input_shape)
    x.append(inputs)
    x.append(SpectralNormalization(TensorProduct(n_dense_filters, kernel_initializer=init,
                           kernel_regularizer=l2(reg)))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))
    x.append(SpectralNormalization(TensorProduct(n_features, kernel_initializer=init,
                           kernel_regularizer=l2(reg)))(x[-1]))
    outputs = tf.expand_dims(x[-1], 0)
    outputs = Reshape((-1,2))(outputs)
    outputs = tf.squeeze(outputs, 0)
    
    outputs, covmat = tfm.nlp.layers.RandomFeatureGaussianProcess(units=n_features,
                                                          normalize_input=True,
                                                          scale_random_features=False,
                                                          return_gp_cov=True)(outputs)

    outputs = tf.expand_dims(outputs, 0)
    outputs = Reshape((-1,128,128,2))(outputs)
    outputs = tf.squeeze(outputs, 0)

    return Model(inputs=inputs, outputs=outputs, name=name)


def offset_regression_head(input_shape,
                           regression_feature_size=256,
                           name='offset_regression_head'):
    """Creates a offset regression head.

    Args:
        input_shape (tuple): Shape of input image.
        regression_feature_size(int)

    Returns:
        tensorflow.keras.Model for offset regression.
    """

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

    outputs = Conv2D(filters=2, name='offset_regression_output', **options)(outputs)

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
               sngp=False,
               **kwargs):
    """Creates a 2D featurenet with prediction heads for spot detection.

    Model architecture based on ``deepcell.model_zoo.bn_feature_net_skip_2D``.

    Args:
        receptive_field (int): the receptive field of the neural network.
        input_shape (tuple): Shape of input image.
        inputs (tensor): optional input tensor
        n_skips (int): The number of skip-connections.
        norm_method (str): Normalization method to use with the
            :mod:``deepcell.layers.normalization.ImageNormalization2D`` layer.
        padding_mode (str): Type of padding, one of `('reflect' or 'zero')`.
        kwargs (dict): Other model options defined in ``~bn_feature_net_2D``.

    Returns:
        tensorflow.keras.Model: 2D FeatureNet with prediction heads for spot
        detection.
    """

    inputs = Input(shape=input_shape)

    featurenet_model = bn_feature_net_skip_2D_sngp(
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
