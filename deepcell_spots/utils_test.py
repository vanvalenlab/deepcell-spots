import scipy
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.affine_transformations import transform_matrix_offset_center
from tensorflow.python.platform import test

from utils import *


class TestUtils(test.TestCase):
    def test_subpixel_distance_transform(self):
        # test shape of output for square image
        point_list = np.random.random((10, 2))
        image_shape = (30, 30)
        delta_y, delta_x, nearest_point = subpixel_distance_transform(
            point_list, image_shape)

        self.assertEqual(np.shape(delta_y), image_shape)
        self.assertEqual(np.shape(delta_x), image_shape)
        self.assertEqual(np.shape(nearest_point), image_shape)

        # test shape of output for rectangular image
        point_list = np.random.random((10, 2))
        image_shape = (50, 30)
        delta_y, delta_x, nearest_point = subpixel_distance_transform(
            point_list, image_shape)

        self.assertEqual(np.shape(delta_y), image_shape)
        self.assertEqual(np.shape(delta_x), image_shape)
        self.assertEqual(np.shape(nearest_point), image_shape)

        # are more tests needed?

    # def test_generate_transformation_matrix(self):
    #     # not sure what to test beside the shape


if __name__ == '__main__':
    test.main()
