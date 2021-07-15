import numpy as np
from skimage.feature import peak_local_max
from skimage import measure
from tensorflow.python.platform import test

from postprocessing_utils import *


class TestPostProcUtils(test.TestCase):
    def test_y_annotations_to_points_list(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, 0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output does not need to be below some threshold value (0.5), regression
        # adds to coords
        num_images = 10
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, :, 1, 1, 1] = np.ones(num_images)
        y_pred[0, :, :, :, :] = np.ones(
            (1, num_images, image_dim, image_dim, 2))
        threshold = 0.9
        coords = y_annotations_to_point_list(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 2)

    def test_y_annotations_to_points_list_restrictive(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, 0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list_restrictive(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output needs to be below some threshold value (0.5)
        num_images = 1
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, 0, 1, 1, 1] = 1
        y_pred[0, :, :, :, :] = np.ones(
            (1, num_images, image_dim, image_dim, 2))
        threshold = 0.9

        coords = y_annotations_to_point_list_restrictive(y_pred, threshold)

        for i in range(len(coords)):
            self.assertEqual(len(coords[i]), 0)

        # Regression output needs to be below some threshold value (0.5), regression adds to coords
        num_images = 10
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, :, 1, 1, 1] = np.ones(num_images)
        y_pred[0, :, :, :, :] = np.ones(
            (1, num_images, image_dim, image_dim, 2))*0.4
        threshold = 0.9
        coords = y_annotations_to_point_list(y_pred, threshold)

        print(coords[0][0])
        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1.4)

    def test_y_annotations_to_point_list_max(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, 0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list_max(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output does not need to be below some threshold value (0.5),
        # regression adds to coords
        num_images = 10
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, :, 1, 1, 1] = np.ones(num_images)
        y_pred[0, :, :, :, :] = np.ones(
            (1, num_images, image_dim, image_dim, 2))
        threshold = 0.9
        coords = y_annotations_to_point_list_max(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 2)

    def test_y_annotations_to_point_list_cc(self):
        # Easy example with one spot
        num_images = 1
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, 0, 1, 1, 1] = 1
        threshold = 0.9

        coords = y_annotations_to_point_list_cc(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 1)

        # Regression output does not need to be below some threshold value (0.5),
        # regression adds to coords
        num_images = 10
        image_dim = 10
        y_pred = np.zeros((2, num_images, image_dim, image_dim, 2))
        y_pred[1, :, 1, 1, 1] = np.ones(num_images)
        y_pred[0, :, :, :, :] = np.ones(
            (1, num_images, image_dim, image_dim, 2))
        threshold = 0.9
        coords = y_annotations_to_point_list_cc(y_pred, threshold)

        for i in range(len(coords)):
            for ii in range(len(coords[i])):
                for dim in range(2):
                    self.assertEqual(coords[i][ii][dim], 2)


if __name__ == '__main__':
    test.main()
