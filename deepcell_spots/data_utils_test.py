import numpy as np
import random
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test
from sklearn.model_selection import train_test_split

from data_utils import *

class TestDataUtils(test.TestCase):
    def test_slice_image(self):
        
        # test square image with square, divisible reshape size
        img_w, img_h = 30, 30
        X = np.random.random((10, img_w, img_h, 1))
        slice_size = [5,5]
        num_slices_x = np.ceil(img_w/slice_size[0])
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X = slice_image(X, slice_size)
        
        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))

        # test square image with rectangular, divisible reshape size
        img_w, img_h = 30, 30
        X = np.random.random((10, img_w, img_h, 1))
        slice_size = [5,6]
        num_slices_x = np.ceil(img_w/slice_size[0])
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X = slice_image(X, slice_size)
        
        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))

        # test rectangular image with square, divisible reshape size
        img_w, img_h = 25,25
        X = np.random.random((10, img_w, img_h, 1))
        slice_size = [5,5]
        num_slices_x = np.ceil(img_w/slice_size[0])
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X = slice_image(X, slice_size)
        
        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))

        # test rectangular image with rectangular, divisible reshape size
        img_w, img_h = 25,30
        X = np.random.random((10, img_w, img_h, 1))
        slice_size = [5,6]
        num_slices_x = np.ceil(img_w/slice_size[0])
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X = slice_image(X, slice_size)
        
        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))

        # test square image with square, indivisible reshape size
        img_w, img_h = 30,30
        X = np.random.random((10, img_w, img_h, 1))
        slice_size = [8,8]
        num_slices_x = np.ceil(img_w/slice_size[0]).astype(int)
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X = slice_image(X, slice_size)
        
        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))
        self.assertEqual((slice_X[num_slices_x-1,:,-1*slice_size[0]*num_slices_x:]).all(),(slice_X[num_slices_x,:,:slice_size[0]*num_slices_x]).all())

    def test_get_data(self):
        test_size = .1
        img_w, img_h = 30, 30
        X = np.random.random((10, img_w, img_h, 1))
        y = np.random.randint(3, size=(10, img_w, img_h, 1))

        # test good filepath
        temp_dir = self.get_temp_dir()
        good_file = os.path.join(temp_dir, 'good.npz')
        np.savez(good_file, X=X, y=y)

        train_dict, test_dict = get_data(
            good_file, test_size=test_size)

        X_test, X_train = test_dict['X'], train_dict['X']

        self.assertIsInstance(train_dict, dict)
        self.assertIsInstance(test_dict, dict)
        self.assertAlmostEqual(X_test.size / (X_test.size + X_train.size), test_size)

        # test bad filepath
        bad_file = os.path.join(temp_dir, 'bad.npz')
        np.savez(bad_file, X_bad=X, y_bad=y)
        with self.assertRaises(KeyError):
            _, _ = get_data(bad_file)

test.main()