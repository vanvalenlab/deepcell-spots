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
"""Dataset builders for deepcell spots"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from keras_preprocessing.image.affine_transformations import apply_affine_transform as affine_transform_image
from keras_preprocessing.image.affine_transformations import flip_axis
from deepcell_spots.utils import affine_transform_points

class SpotDatasetBuilder(object):
    def __init__(self,
                 train_dict,
                 min_spots = 3,
                 batch_size=1,
                 cache_size=64,
                 augmentation_kwargs={'rotation_range':180,
                                      'zoom_range':(0.5, 2),
                                      'horizontal_flip': True,
                                      'vertical_flip': True}):

        # Load training data
        X = train_dict['X']

        # TODO: Figure out how to deal with varying numbers of points
        # for each image - right now, just treating it as equal
        # Solution is likely ragged tensor

        y = train_dict['y']
        tpr = train_dict['tpr']
        fpr = train_dict['fpr']
        sigma = train_dict['sigma']

        self.X = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype=K.floatx())
        self.tpr = tpr
        self.fpr = fpr
        self.sigma = sigma
        self.augmentation_kwargs = augmentation_kwargs

        self.batch_size = batch_size
        self.cache_size = cache_size

        # Create dataset
        self._create_dataset()
        
    def _transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]], dtype='float32')
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]], dtype='float32')
        
        offset_matrix = tf.convert_to_tensor(offset_matrix)
        reset_matrix = tf.convert_to_tensor(reset_matrix)
        
        transform_matrix = tf.keras.backend.dot(tf.keras.backend.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix
        
    def _compute_random_transform_matrix(self):
        rotation_range = self.augmentation_kwargs['rotation_range']
        zoom_range = self.augmentation_kwargs['zoom_range']
        horizontal_flip = self.augmentation_kwargs['horizontal_flip']
        vertical_flip = self.augmentation_kwargs['vertical_flip']
        
        
        # Get random angles
        theta = tf.random.uniform(shape=(1,), 
                                  minval=-np.pi*rotation_range/180, 
                                  maxval=np.pi*rotation_range/180)
        one = tf.constant(1.0, shape=(1,))
        zero = tf.constant(0.0, shape=(1,))
        cos_theta = tf.math.cos(theta)
        sin_theta = tf.math.sin(theta)
        
        rot_row_0 = tf.stack([cos_theta, -sin_theta, zero], axis=1)
        rot_row_1 = tf.stack([sin_theta, cos_theta, zero], axis=1)
        rot_row_2 = tf.stack([zero, zero, one], axis=1)
        rotation_matrix = tf.concat([rot_row_0, rot_row_1, rot_row_2], axis=0)
        
        transform_matrix = rotation_matrix
        
        # Get random lr flips
        lr = 2*tf.cast(tf.random.categorical(tf.math.log([[0.5, 0.5]]), 1), 'float32')[0] - 1.0
        lr_row_0 = tf.stack([lr, zero, zero], axis=1)
        lr_row_1 = tf.stack([zero, one, zero], axis=1)
        lr_row_2 = tf.stack([zero, zero, one], axis=1)
        lr_flip_matrix = tf.concat([lr_row_0, lr_row_1, lr_row_2], axis=0)
        
        transform_matrix = tf.keras.backend.dot(transform_matrix, lr_flip_matrix)
        
        # Get randum ud flips
        ud = 2*tf.cast(tf.random.categorical(tf.math.log([[0.5, 0.5]]), 1), 'float32')[0] - 1.0
        ud_row_0 = tf.stack([one, zero, zero], axis=1)
        ud_row_1 = tf.stack([zero, ud, zero], axis=1)
        ud_row_2 = tf.stack([zero, zero, one], axis=1)
        ud_flip_matrix = tf.concat([ud_row_0, ud_row_1, ud_row_2], axis=0)
        
        transform_matrix = tf.keras.backend.dot(transform_matrix, ud_flip_matrix)

        # Get random zooms
        zx = tf.random.uniform(shape=(1,), minval=zoom_range[0], maxval=zoom_range[1])
        zy = tf.random.uniform(shape=(1,), minval=zoom_range[0], maxval=zoom_range[1])
        z_row_0 = tf.stack([zx, zero, zero], axis=1)
        z_row_1 = tf.stack([zero, zy, zero], axis=1)
        z_row_2 = tf.stack([zero, zero, one], axis=1)
        zoom_matrix = tf.concat([z_row_0, z_row_1, z_row_2], axis=0)
        
        transform_matrix = tf.keras.backend.dot(transform_matrix, zoom_matrix)

        # Combine all matrices
        h, w = self.X.shape[1], self.X.shape[2]
        transform_matrix = self._transform_matrix_offset_center(transform_matrix, h, w)
        
        return transform_matrix      

    def _augment(self, *args):
        X_dict = args[0]
        y_dict = args[1]

        images = X_dict['images']
        points = y_dict['points']

        transform_matrix = self._compute_random_transform_matrix()
        
        # Transform image
        transform_vector = tfa.image.transform_ops.matrices_to_flat_transforms(transform_matrix)
    
        new_images = tfa.image.transform(images,
                                     transform_vector,
                                     interpolation = 'BILINEAR')
        
        # Transform points
        offsets = transform_matrix[:2, 2]
        tr_matrix = transform_matrix[0:2, 0:2]
        
        # flip x and y coordinates 
        tr_matrix = tf.reverse(tr_matrix, axis=[0,1])
        offsets = tf.reverse(offsets, axis=[0])
        
        inv = tf.linalg.inv(tr_matrix)
        
        # Only transform the good points
        valid_indices = tf.where(points[:,0] != -1)
        new_points = tf.gather(points, valid_indices[:,0], axis=0)
        new_points = tf.matmul(inv, new_points-offsets, transpose_b=True)
        new_points = tf.transpose(new_points)
        
        # Repad
        paddings = tf.convert_to_tensor([[0,tf.shape(points)[0]-tf.shape(new_points)[0]],[0,0]])
        new_points = tf.pad(new_points, paddings, constant_values=-1)
        
        # Update dictionaries
        X_dict['images'] = new_images
        y_dict['points'] = new_points

        return X_dict, y_dict

    def _create_dataset(self):
        X_dict = {}
        X_dict['images'] = self.X
        X_dict['tpr'] = self.tpr
        X_dict['fpr'] = self.fpr
        X_dict['sigma'] = self.sigma

        y_dict = {}
        y_dict['points'] = self.y

        self.dataset = tf.data.Dataset.from_tensor_slices((X_dict, y_dict))

        # Apply augmentation, batching, and caching
        self.training_dataset = self.dataset.shuffle(self.cache_size).map(self._augment).batch(self.batch_size)
