### 
""" CNN architechture with classification and regression outputs for dot center detection"""


import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Concatenate
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Permute, Reshape
from tensorflow.python.keras.layers import Activation, Softmax, Lambda
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2

### for running on my laptop:
import sys
sys.path.append(r'C:\Users\nitza\OneDrive\Documents\GitHub\deepcell-tf')
### end for laptop deepcel import
from deepcell.model_zoo import bn_feature_net_skip_2D
from deepcell.layers import DilatedMaxPool2D, DilatedMaxPool3D
from deepcell.layers import ImageNormalization2D, ImageNormalization3D
from deepcell.layers import Location2D, Location3D
from deepcell.layers import ReflectionPadding2D, ReflectionPadding3D
from deepcell.layers import TensorProduct


def default_heads(input_shape, num_classes):
    """
    Create a list of the default heads for dot detection center pixel detection and offset regression

    Args:
      input_shape
      num_classes

    Returns:
      A list of tuple, where the first element is the name of the submodel
      and the second element is the submodel itself.

    """
    num_dimensions = 2 # regress x and y coordinates (pixel center signed distance from nearest object center)
    return [
      ('offset_regression', offset_regression_head(num_values=num_dimensions, input_shape=input_shape)),
      ('classification', classification_head(input_shape,n_features=num_classes))
    ]


def classification_head(input_shape,
                        n_features=2,
                        n_dense_filters=128,
                        reg=1e-5,
                        init='he_normal',
                        name='classification_head'):
    """
    Creates a classification head

    Args:
        n_features (int): Number of output features (number of possible classes for each pixel. default is 2: contains point / does not contain point)
        reg (int): regularization value
        init (str): Method for initalizing weights.

    Returns:
        tensorflow.keras.Model for classification (softmax output)
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = [] # Create layers list (x) to store all of the layers.
    inputs = Input(shape=input_shape)
    x.append(inputs)
    x.append(TensorProduct(n_dense_filters, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))
    x.append(TensorProduct(n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))
    #x.append(Flatten()(x[-1]))
    outputs = Softmax(axis=channel_axis)(x[-1])
    #x.append(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


def rn_classification_head(num_classes,
                           input_size,
                           input_feature_size=256,
                           prior_probability=0.01,
                           classification_feature_size=256,
                           name='classification_submodel'):
    # similar to the one used by retinanet, HASN'T BEEN TESTED, MAY NOT WORK!!!
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    if K.image_data_format() == 'channels_first':
        inputs = Input(shape=(input_size, None, None))
    else:
        inputs = Input(shape=(None, None, input_size))
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if K.image_data_format() == 'channels_first':
        outputs = Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)

    

def offset_regression_head(num_values,
                           input_shape,
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

# TO BE DELETED - only needed when there are multiple features but here have only 1 backbone_output
def __build_model_heads(name, model, backbone_output):
    identity = Lambda(lambda x: x, name=name)
    return identity(model(backbone_output))
    #for name, model in head_submodels: # DELETE?
    #concat = Concatenate(axis=1, name=name)
    #return concat([model(backbone_output)])


def dot_net_2D(receptive_field=13,
               input_shape=(256, 256, 1),
               inputs=None,
               n_skips=3,
               norm_method='std',
               padding_mode='reflect',
               **kwargs):


    inputs = Input(shape=input_shape)

    #models  = []
    #model_outputs = [] # AAA outputs all the intermediate outputs used for skips in featurenet

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
    #model_outputs.append(featurenet_output)
    #models.append(featurenet_model)

    # add 2 heads: 1 for center pixel classification (should be 1 for pixel which has center, 0 otherwise),
    # and 1 for center location regression (size of classification output where pixel value = signed x/y distance to nearest max of classification)
    # softmax top (as in include_top==True for bn_feature_net_2D):

    input_shape = featurenet_output.get_shape().as_list()[1:]
    
    print('input_shape:', input_shape) # DEBUG

    head_submodels = default_heads(input_shape=input_shape, num_classes=2) # 2 classes: contains / does not contain dot center
    dot_head = [__build_model_heads(n, m, featurenet_output) for n, m in head_submodels]
    outputs = dot_head

    #model = Model(inputs=inputs, outputs=outputs, name=name)
    model = Model(inputs=inputs, outputs=outputs)

    return model
