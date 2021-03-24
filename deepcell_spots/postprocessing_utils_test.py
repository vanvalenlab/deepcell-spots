import numpy as np
from skimage.feature import peak_local_max
from skimage import measure
from tensorflow.python.platform import test

from postprocessing_utils import *

class TestUtils(test.TestCase):
    def test_y_annotations_to_points_list(self):
        num_images = 10
        image_dim = 128
        y_pred = np.random.random_sample((2,num_images,image_dim,image_dim,2))
        ind = 0
        threshold = 0.9

test.main()