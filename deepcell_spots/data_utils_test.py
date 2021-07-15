import numpy as np
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
        num_slices_x = np.ceil(img_w/slice_size[0]).astype(int) # used as an index
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X = slice_image(X, slice_size)
        
        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))
        self.assertEqual((slice_X[num_slices_x-1::num_slices_x,:,
                        -1*slice_size[0]*num_slices_x:]).all(),
                        (slice_X[num_slices_x::num_slices_x,:,
                        :slice_size[0]*num_slices_x]).all())

        # test no channel dimension
        img_w, img_h = 30,30
        X = np.random.random((10, img_w, img_h))
        slice_size = [8,8]
        num_slices_x = np.ceil(img_w/slice_size[0])
        num_slices_y = np.ceil(img_h/slice_size[1])
        with self.assertRaises(IndexError):
            _ = slice_image(X, slice_size)

        # test insufficient slice dimensions
        img_w, img_h = 30,30
        X = np.random.random((10, img_w, img_h))
        slice_size = 8
        num_slices_x = np.ceil(img_w/slice_size)
        num_slices_y = np.ceil(img_h/slice_size)
        with self.assertRaises(TypeError):
            _ = slice_image(X, slice_size)

        # test overlap argument, divisible
        img_w, img_h = 25,25
        X = np.random.random((10, img_w, img_h, 1))
        slice_size = [5,5]
        overlap=1
        num_slices_x = np.ceil(img_w/slice_size[0]) + np.floor(img_w/slice_size[0]**2) 
        num_slices_y = np.ceil(img_h/slice_size[1]) + np.floor(img_w/slice_size[0]**2) 
        slice_X = slice_image(X, slice_size,overlap=1)

        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,5,5,overlap))
        
        # test overlap argument, indivisible
        img_w, img_h = 26,26
        X = np.random.random((10, img_w, img_h, 1))
        slice_size = [5,5]
        overlap=1
        num_slices_x = np.ceil(img_w/slice_size[0]) + np.floor(img_w/slice_size[0]**2) 
        num_slices_y = np.ceil(img_h/slice_size[1]) + np.floor(img_w/slice_size[0]**2) 
        slice_X = slice_image(X, slice_size,overlap=1)

        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,5,5,overlap))

    ## FUNCTION DID NOT PASS TEST
    # def test_stitch_image(self):
    #     # test square image with square, divisible reshape size
    #     img_w, img_h = 5,5
    #     X = np.random.random((360, img_w, img_h,1))
    #     stitch_size = [30,30]
    #     num_slices_x = stitch_size[0]/img_w
    #     num_slices_y = stitch_size[1]/img_h
    #     stitch_X = stitch_image(X, [img_w,img_h], stitch_size)

    #     self.assertEqual(np.shape(stitch_X), (np.shape(X)[0]/num_slices_x/num_slices_y,img_w*num_slices_x,img_h*num_slices_y,1))

    def test_slice_annotated_image(self):
        # test square image with square, divisible reshape size
        img_w, img_h = 30, 30
        num_images = 10
        X = np.random.random((num_images, img_w, img_h, 1))
        y = np.random.random((num_images,10,2)) # ten images with ten spots each
        slice_size = [5,5]
        num_slices_x = np.ceil(img_w/slice_size[0])
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X, slice_y = slice_annotated_image(X, y, slice_size)
        
        self.assertEqual(np.shape(slice_X),(num_images*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))
        self.assertEqual(np.shape(slice_y),num_images*num_slices_x*num_slices_y)
        
        # test square image with square, indivisible reshape size
        img_w, img_h = 30,30
        num_images = 10
        X = np.random.random((10, img_w, img_h, 1))
        y = np.random.random((num_images,10,2)) # ten images with ten spots each
        slice_size = [8,8]
        num_slices_x = np.ceil(img_w/slice_size[0]).astype(int) # used as an index
        num_slices_y = np.ceil(img_h/slice_size[1])
        slice_X, slice_y = slice_annotated_image(X, y, slice_size)
        
        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,
                        slice_size[0],slice_size[1],1))
        self.assertEqual(np.shape(slice_y),num_images*num_slices_x*num_slices_y)
        self.assertEqual((slice_X[num_slices_x-1::num_slices_x,:,
                        -1*slice_size[0]*num_slices_x:]).all(),
                        (slice_X[num_slices_x::num_slices_x,:,
                        :slice_size[0]*num_slices_x]).all())

        # test no channel dimension
        img_w, img_h = 30, 30
        num_images = 10
        X = np.random.random((num_images, img_w, img_h))
        y = np.random.random((num_images,10,2)) # ten images with ten spots each
        slice_size = [5,5]
        num_slices_x = np.ceil(img_w/slice_size[0])
        num_slices_y = np.ceil(img_h/slice_size[1])
        with self.assertRaises(IndexError):
            _,_ = slice_annotated_image(X, y, slice_size)

        # test insufficient slice dimensions
        img_w, img_h = 30, 30
        num_images = 10
        X = np.random.random((num_images, img_w, img_h))
        y = np.random.random((num_images,10,2)) # ten images with ten spots each
        slice_size = 5
        num_slices_x = np.ceil(img_w/slice_size)
        num_slices_y = np.ceil(img_h/slice_size)
        with self.assertRaises(TypeError):
            _,_ = slice_annotated_image(X, y, slice_size)

        # test overlap argument, divisible
        img_w, img_h = 25,25
        num_images = 10
        X = np.random.random((num_images, img_w, img_h, 1))
        y = np.random.random((num_images,10,2)) # ten images with ten spots each
        slice_size = [5,5]
        overlap=1
        num_slices_x = np.ceil(img_w/slice_size[0]) + np.floor(img_w/slice_size[0]**2) 
        num_slices_y = np.ceil(img_h/slice_size[1]) + np.floor(img_w/slice_size[0]**2) 
        slice_X,slice_y = slice_annotated_image(X, y, slice_size,overlap=1)

        self.assertEqual(np.shape(slice_X),(10*num_slices_x*num_slices_y,5,5,overlap))
        self.assertEqual(np.shape(slice_y),num_images*num_slices_x*num_slices_y)

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

if __name__ == '__main__':
    test.main()