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
"""Layers for hybrid CNN-probabalistic models for spot detection"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from tensorflow_probability import distributions as tfd


class ImageToCoords(Layer):
	def __init__(self, 
				 parallel_iterations=32,
				 *args,
				 **kwargs):
	self.parallel_iterations = parallel_iterations
	super(ImageToCoords, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		detections = inputs[0]
		deltas = inputs[1]

		# Get the maximum number of detections in 
		# any batch
		nonzero = tf.math.count_nonzero(detections, axis=[1,2,3])
		max_det = tf.math.reduce_max(nonzero)

		# Create a function to compute the coordinates for a batch
		def _get_coords(args):
			detections = args[0]
			deltas = args[1]

			coords = tf.where(tf.math.equal(detections, 1))
			gathered_deltas = tf.gather_nd(deltas, coords)

			subpixel_coords = coords + gathered_deltas

			# Pad subpixel_coords
			paddings = tf.convert_to_tensor([[0, max_det-tf.shape(coords)[0]], [0, 0]])
			padded_coords = tf.pad(subpixel_coords,
								   paddings,
								   mode='CONSTANT',
								   constant_values=-1)
			return padded_coords

		coords_batch = tf.map_fn(_get_coords,
								 elems=[detections, deltas],
								 dtype=K.floatx(),
								 parallel_iterations=self.parallel_iterations)		

		return coords_batch

	def compute_output_shape(self, input_shape)
		output_shape = (None, None, 2)
		return tensor_shape.TensorShape(output_shape)

	def get_config(self):
		config = {
			'parallel_iterations': self.parallel_iterations
		}
		base_config = super(ImageToCoords, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))	


class AnnotatorDetection(Layer):
	def __init__(self, 
				 *args, 
				 **kwargs):
	super(AnnotatorDetection, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		pixel_probs = inputs[0]
		true_pos_rate = inputs[1]
		false_pos_rate = inputs[2]

		detection_probs = pixel_probs * true_pos_rate + (1-pixel_probs) * false_pos_rate
		bernoulli = tfd.Bernoulli(probs=detection_probs)

		return bernoulli

	def compute_output_shape(self, input_shape):
		output_shape = input_shape[0]

		return tensor_shape.TensorShape(output_shape)

	def get_config(self):
		base_config = super(AnnotatorDetection, self).get_config()
		return base_config


class AnnotatorLocalizationError(Layer):
	def __init__(self,
		 		 *args,
		 		 **kwargs):
	super(AnnotatorLocalizationError).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		coords = inputs[0]
		sigma = inputs[1]

		scale = sigma * tf.ones(tf.shape(coords))
		normal = tfd.MultivariateNormalDiag(loc=coords, scale=scale)

		return normal

	def compute_output_shape(self, input_shape):
		output_shape = input_shape[0]

		return tensor_shape.TensorShape(output_shape)

	def get_config(self):
		base_config = super(AnnotatorLocalizationError, self)
		return base_config

