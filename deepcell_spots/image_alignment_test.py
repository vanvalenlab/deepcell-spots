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

"""Tests for loading and aligning images"""

import numpy as np
from tensorflow.python.platform import test

from deepcell_spots.image_alignment import crop_images, align_images


def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value


class TestImageAlignment(test.TestCase):
    def test_align_images(self):
        im_size = 100
        shift = 5

        im_dict = {}
        ref_dict = {}
        for i in range(10):
            im = np.random.random((1, im_size, im_size, 1)) * 65535
            shift_im = np.expand_dims(np.pad(im[0, ..., 0],
                                      shift,
                                      pad_with)[:im_size, :im_size], axis=[0, -1])

            im_dict[i] = im
            ref_dict[i] = shift_im
        aligned_dict = align_images(im_dict, ref_dict)

        self.assertEqual(np.shape(im_dict.values()), np.shape(aligned_dict.values()))

    def test_crop_images(self):
        # With padding
        im = np.random.random((8, 8)) + 0.1
        im = np.pad(im, [(1, 1), (1, 1)], mode='constant')
        im = np.expand_dims(im, axis=[0, -1])
        aligned_dict = {0: im}

        crop_dict = crop_images(aligned_dict)

        self.assertEqual(list(crop_dict.keys()), [0])
        self.assertEqual(np.shape(crop_dict[0]), (1, 6, 6, 1))

        # Without padding
        im = np.random.random((8, 8)) + 0.1
        im = np.expand_dims(im, axis=[0, -1])
        aligned_dict = {0: im}

        crop_dict = crop_images(aligned_dict)

        self.assertEqual(list(crop_dict.keys()), [0])
        self.assertEqual(np.shape(crop_dict[0]), (1, 6, 6, 1))


if __name__ == '__main__':
    test.main()
