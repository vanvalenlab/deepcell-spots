# Copyright 2019-2023 The Van Valen Lab at the California Institute of
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

"""Tests for spot detection image generators"""

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.platform import test

from deepcell_spots import image_generators


def all_test_images():
    img_w = img_h = 20
    rgb_images = []
    rgba_images = []
    gray_images = []
    for n in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        rgb_images.append(im)

        imarray = np.random.rand(img_w, img_h, 4) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        rgba_images.append(im)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = Image.fromarray(
            imarray.astype('uint8').squeeze()).convert('L')
        gray_images.append(im)

    return [gray_images]


class TestImageFullyConvDotDataGenerator(test.TestCase):
    def test_image_fully_conv_dot_data_generator(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.ImageFullyConvDotDataGenerator(
                rotation_range=0,
                shear_range=0,
                zoom_range=0,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                cval=0.
            )

            img_w, img_h = 21, 21
            test_batches = 8

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, img_w, img_h, 1)),
                'y': np.random.randint(img_h, size=(test_batches, 10, 2))
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(img_h, size=(test_batches, 10, 2))
            for x, _ in generator.flow(
                    train_dict,
                    batch_size=1,
                    shuffle=True):
                self.assertEqual(x.shape[1:], (20, 20, 1))
                break

    def test_sample_data_generator_invalid_data(self):
        generator = image_generators.ImageFullyConvDotDataGenerator(
            rotation_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            cval=0.)

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((8, 10, 10)),
                'y': np.random.random((8, 10, 10))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((7, 10, 10, 1))
            }
            generator.flow(train_dict)

        with self.assertRaises(ValueError):
            generator = image_generators.ImageFullyConvDotDataGenerator(
                data_format='unknown')

        with self.assertRaises(ValueError):
            generator = image_generators.ImageFullyConvDotDataGenerator(
                zoom_range=(2, 2, 2))
