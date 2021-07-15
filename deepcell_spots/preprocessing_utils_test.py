import random
import numpy as np

from tensorflow.python.platform import test

from preprocessing_utils import *


class TestPreProcUtils(test.TestCase):
    def test_mean_std_normalize(self):
        image_dims = 128
        image = np.random.random((2, image_dims, image_dims, 1))
        norm_image = mean_std_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)

        # test convert to int
        image_dims = 128
        image = np.ones((2, image_dims, image_dims, 1)).astype(int)
        norm_image = mean_std_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)

    def test_min_max_normalize(self):
        image_dims = 128
        image = np.random.random((2, image_dims, image_dims, 1))
        norm_image = min_max_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)

        # test convert to int
        image_dims = 128
        image = np.ones((2, image_dims, image_dims, 1)).astype(int)
        norm_image = min_max_normalize(image)

        self.assertEqual(image.shape, norm_image.shape)


if __name__ == '__main__':
    test.main()
