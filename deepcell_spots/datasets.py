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
	self._create_dataset(self)

	def _get_random_transform(self, seed=None):
		rotation_range = self.augmentation_kwargs['rotation_range']
		zoom_range = self.augmentation_kwargs['zoom_range']
		horizontal_flip = self.augmentation_kwargs['horizontal_flip']
		vertical_flip = self.augmentation_kwargs['vertical_flip']

		theta = np.random.uniform(-rotation_range, rotation_range)
		
		if zoom_range[0] == 1 and zoom_range[1] == 1:
			zx, zy = 1, 1
		else:
			zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

		flip_horizontal = (np.random.random() < 0.5) * horizontal_flip
		flip_vertical = (np.random.random() < 0.5) * vertical_flip

		transform_parameters = {'theta': theta,
								'zx': zx,
								'zy': zy,
								'flip_horizontal': flip_horizontal,
								'flip_vertical': flip_vertical,
								'tx': 0,
								'ty': 0,
								'shear':0,
								'channel_shift_intensity': None,
								'brightness': None}
		return transform_parameters

	def _augment_data(self, images, points):
		# Get transformation matrix
		transform_parameters = self._get_random_transform()

		# Apply transform to image
		transformed_image = affine_transform_image(image,
												   transform_parameters.get('theta', 0),
												   transform_parameters.get('tx', 0),
												   transform_parameters.get('ty', 0),
												   transform_parameters.get('shear', 0),
												   transform_parameters.get('zx', 1),
												   transform_parameters.get('zy', 1),
												   row_axis=0,
												   col_axis=1,
												   channel_axis=2,
												   fill_mode='nearest',
												   cval=0,
												   order=1)

		if transform_parameters.get('flip_horizontal', False):
			transformed_image = flip_axis(transformed_image, 0)

		if transformed_parameters.get('flip_vertical', False):
			transformed_image = flip_axis(transformed_image, 1)

		# Apply transform to points
		transformed_points = affine_transform_points(points, 
													 transform_parameters, 
													 image_shape=self.X.shape[1:])

		return transformed_image, transformed_points

	def _augment(self, args):
		X_dict = args[0]
		y_dict = args[1]

		images = X_dict['images']
		points = y_dict['points']

		im_shape = images.shape
		pt_shape = points.shape

		[images, points] = tf.py_function(self._augment_data, [images, points], [tf.float32, tf.float32])

		images.set_shape(im_shape)
		points.set_shape(pt_shape)

		X_dict['images'] = images
		y_dict['points'] = points

		return X_dict, y_dict

	def _create_dataset(self):
		X_dict = {}
		X_dict['images'] = self.X
		X_dict['tpr'] = self.tpr
		X_dict['fpr'] = self.fpr
		X_dict['sigma'] = self.sigma

		y_dict = {}
		y_dict['points'] = tf.ragged.constant(self.y)

		self.dataset = tf.data.Dataset.from_tensor_slices((X_dict, y_dict))

		# Apply augmentation, batching, and caching
		self.training_dataset = self.dataset.shuffle(self.cache_size).map(self._augment).padded_batch(self.batch_size, padding_values=-1)

