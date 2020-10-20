# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/tf-keras-retinanet/LICENSE
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
"""Model zoo for deepcell spots"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, 

class SpotsModel(object):
    def __init__(self, 
                 image_shape=(256,256,1),
                 receptive_field=13,
                 norm_method='std',
                 padding_mode='reflect'):
        self.image_shape = image_shape
        self.receptive_field = receptive_field
        self.norm_method = norm_method
        self.padding_mode = padding_mode

    def _create_model(self):

        # Create image encoder
        images = Input(shape=self.image_shape,
                       name='images')

        self._create_image_encoder()
        encoded = self.image_encoder(images)
        pixel_probs, deltas = encoded.outputs

        # Apply annotator detection model
        tpr = Input(shape=(1,),
                    name='tpr')
        fpr = Input(shape=(1,),
                    name='fpr')
        detections = AnnotatorDetection()([pixel_probs, tpr, fpr])

        # Convert encoded image to coordinates
        coords = ImageToCoords()([detections, deltas])

        # Apply annotator localization error model
        sigma = Input(shape=(1,),
                      name='sigma')

        spots = AnnotatorLocalization()([coords, sigma])

        self.model = Model(inputs=[images, tpr, fpr, sigma], outputs=spots)

    def _create_image_encoder(self):

        inputs = Input(shape=self.input_shape)

        def _classification_head(input_shape,
                                 n_features=1,
                                 n_dense_filters=128,
                                 name='classification_head'):
            
            inputs = Input(shape=input_shape)
            x = inputs
            x = Dense(n_dense_filters)(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = Dense(n_features)
            x = Activation('sigmoid')(x)

            return Model(inputs=inputs, outputs=x, name=name)

        def _regression_head(input_shape,
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

        backbone = bn_featurenet_2D(receptive_field=self.receptive_field,
                                    input_shape=self.input_shape,
                                    norm_method=self.norm_method,
                                    padding_mode=self.padding_mode,
                                    last_only=True,
                                    include_top=False)
        backbone_output = backbone(inputs)
        classification = _classification_head(self.input_shape)(backbone_output)
        regression = _regression_head(self.input_shape)(backbone_output)
        outputs = [classification, regression]
        
        self.image_encoder = Model(inputs=inputs, outputs=outputs)




    