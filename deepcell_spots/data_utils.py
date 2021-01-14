import numpy as np
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split


def slice_image(X, reshape_size, overlap=0):
    ''' 
    Slice images in X into smaller parts. similar to deepcell.utils.data_utils reshape_matrix
    
    Args:
        X (np.array) containing images: has size (img_number, y, x, channel)
        reshape_size: list of 2 values: y_size, x_size
        overlap (int): number of pixels overlapping in each row/column with the pixels from the same row/column in the neighboring slice
        
    Returns:
        new_X: stack of reshaped images in order of small to large y, then small to large x position in the original image
        np.array of size (n*img_number, y_size, x_size, channel)
        where n = number of images each image in X was sliced into
        if the original image lengths aren't divisible by y_size, x_size, the last image in each row / column overlaps with the one before
    '''
    image_size_x = X.shape[1]
    image_size_y = X.shape[2]
    
    L_x = reshape_size[0] # x length of each slice
    L_y = reshape_size[1] # y length of each slice
    
    n_x = np.int(np.ceil((image_size_x - 2*L_x + overlap)/(L_x - overlap)) + 2) # number of slices along x axis
    n_y = np.int(np.ceil((image_size_y - 2*L_y + overlap)/(L_y - overlap)) + 2) # number of slices along y axis
    
    new_batch_size = X.shape[0] * n_x * n_y # number of images in output
    
    new_X_shape = (new_batch_size, L_x, L_y, X.shape[3])
    new_X = np.zeros(new_X_shape, dtype=K.floatx())

    counter = 0
    for b in range(X.shape[0]):
        for i in range(n_x):
            for j in range(n_y):
                _axis = 1
                if i != n_x - 1:
                    x_start, x_end = i * (L_x - overlap), i * (L_x - overlap) + L_x
                else:
                    x_start, x_end = -L_x, X.shape[_axis]
                    
                if j != n_y - 1:
                    y_start, y_end = j * (L_y - overlap), j * (L_y - overlap) + L_y
                else:
                    y_start, y_end = -L_y, X.shape[_axis + 1]

                new_X[counter] = X[b, x_start:x_end, y_start:y_end, :]
                counter += 1

    print('Sliced data from {} to {}'.format(X.shape, new_X.shape))
    return new_X  
    
    
def stitch_image(X_sliced, image_size, reshape_size, overlap=0):
    '''stitch an image from pieces as those created by slice_image(X, reshape_size, overlap)
    pixels from overlapping region will be taken from the slice where they are farthest from the edge'''
    
    image_size_x = image_size[0]
    image_size_y = image_size[1]
    
    L_x = reshape_size[0] # x length of each slice
    L_y = reshape_size[1] # y length of each slice
    
    n_x = np.int(np.ceil((image_size_x - 2*L_x + overlap)/(L_x - overlap)) + 2) # number of slices along x axis
    n_y = np.int(np.ceil((image_size_y - 2*L_y + overlap)/(L_y - overlap)) + 2) # number of slices along y axis
    
    stitched_batch_size = np.int(X_sliced.shape[0] / (n_x * n_y)) # number of images in stitched output
    
    stitched_X_shape = (stitched_batch_size, image_size_x, image_size_y, X_sliced.shape[3])
    stitched_X = np.zeros(stitched_X_shape, dtype=K.floatx())

    counter = 0
    for b in range(stitched_batch_size):
        for i in range(n_x):
            for j in range(n_y):
                _axis = 1
                if i==0:
                    x_start, x_end = 0, L_x - np.floor(overlap/2)
                    slice_x_start, slice_x_end = 0, L_x - np.floor(overlap/2)
                elif i != n_x - 1:
                    x_start, x_end = i * (L_x - overlap) + np.ceil(overlap/2), i * (L_x - overlap) + L_x - np.floor(overlap/2)
                    slice_x_start, slice_x_end = np.ceil(overlap/2), L_x - np.floor(overlap/2)
                else:
                    last_overlap = (n_x - 2) * (L_x - overlap) + L_x - (image_size_x - L_x)
                    x_start, x_end = -L_x + np.ceil(last_overlap/2), image_size_x
                    slice_x_start, slice_x_end = np.ceil(last_overlap/2), L_x

                if j==0:
                    y_start, y_end = 0, L_y - np.floor(overlap/2)
                    slice_y_start, slice_y_end = 0, L_y - np.floor(overlap/2)
                elif j != n_y - 1:
                    y_start, y_end = j * (L_y - overlap) + np.ceil(overlap/2), j * (L_y - overlap) + L_y - np.floor(overlap/2)
                    slice_y_start, slice_y_end = np.ceil(overlap/2), L_y - np.floor(overlap/2)
                else:
                    last_overlap = (n_y - 2) * (L_y - overlap) + L_y - (image_size_y - L_y)
                    y_start, y_end = -L_y + np.ceil(last_overlap/2), image_size_y
                    slice_y_start, slice_y_end = np.ceil(last_overlap/2), L_y
                
                # cast all indices as integers
                x_start = np.int(x_start)
                x_end = np.int(x_end)
                slice_x_start = np.int(slice_x_start)
                slice_x_end = np.int(slice_x_end)
                y_start = np.int(y_start)
                y_end = np.int(y_end)
                slice_y_start = np.int(slice_y_start)
                slice_y_end = np.int(slice_y_end)
                stitched_X[b, x_start:x_end, y_start:y_end, :] = X_sliced[counter,slice_x_start:slice_x_end,slice_y_start:slice_y_end,:]
                counter += 1

    print('Stitched data from {} to {}'.format(X_sliced.shape, stitched_X.shape))
    return stitched_X


def slice_annotated_image(X, y, reshape_size, overlap=0):
    '''
    Slice images in X into smaller parts. similar to deepcell.utils.data_utils reshape_matrix

    Args:
        X (np.array) containing images: has shape (img_number, y, x, channel)
        reshape_size: list of 2 values: y_size, x_size
        overlap (int): number of pixels overlapping in each row/column with the pixels from the same row/column in the neighboring slice
        y (list / np.array) containing coordinate annotations: has length (img_number),
        each element of the list is a (N,2) np.array where N=the number of points in the image

    Returns:
        new_X: stack of reshaped images in order of small to large y, then small to large x position in the original image
        np.array of size (n*img_number, y_size, x_size, channel)
        where n = number of images each image in X was sliced into
        if the original image lengths aren't divisible by y_size, x_size, the last image in each row / column overlaps with the one before

        new_y: list of length n*img_number
    '''
    image_size_y = X.shape[1]
    image_size_x = X.shape[2]

    L_y = reshape_size[0]  # y length of each slice
    L_x = reshape_size[1]  # x length of each slice

    n_y = np.int(np.ceil((image_size_y - 2 * L_y + overlap) / (L_y - overlap)) + 2)  # number of slices along y axis
    n_x = np.int(np.ceil((image_size_x - 2 * L_x + overlap) / (L_x - overlap)) + 2)  # number of slices along x axis

    new_batch_size = X.shape[0] * n_y * n_x  # number of images in output

    new_X_shape = (new_batch_size, L_y, L_x, X.shape[3])
    new_X = np.zeros(new_X_shape, dtype=K.floatx())

    new_y = [None] * new_batch_size

    counter = 0
    for b in range(X.shape[0]):
        for i in range(n_y):
            for j in range(n_x):
                _axis = 1
                if i != n_y - 1:
                    y_start, y_end = i * (L_y - overlap), i * (L_y - overlap) + L_y
                else:
                    y_start, y_end = X.shape[_axis]-L_y, X.shape[_axis]

                if j != n_x - 1:
                    x_start, x_end = j * (L_x - overlap), j * (L_x - overlap) + L_x
                else:
                    x_start, x_end = X.shape[_axis + 1]-L_x, X.shape[_axis + 1]

                new_X[counter] = X[b, y_start:y_end, x_start:x_end, :]

                new_y[counter] = np.array(
                    [[y0-y_start, x0-x_start] for y0, x0 in y[b] if (y_start-0.5) <= y0 < (y_end-0.5) and
                     (x_start-0.5) <= x0 < (x_end-0.5)])

                counter += 1

    print('Sliced data from {} to {}'.format(X.shape, new_X.shape))
    return new_X, new_y


def get_data(file_name, test_size=.2, seed=0, allow_pickle=False):
    """Load data from NPZ file and split into train and test sets
    This is a copy of deepcell's utils.data_utils.get_data, with allow_pickle added and mode removed

    Args:
        file_name (str): path to NPZ file to load
        test_size (float): percent of data to leave as testing holdout
        seed (int): seed number for random train/test split repeatability
        allow_pickle (bool): if True, allow loading pickled object arrays stored in npz files (via numpy.load)

    Returns:
        (dict, dict): dict of training data, and a dict of testing data
    """

    training_data = np.load(file_name, allow_pickle=allow_pickle)
    X = training_data['X']
    y = training_data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    train_dict = {
        'X': X_train,
        'y': y_train
    }

    test_dict = {
        'X': X_test,
        'y': y_test
    }

    return train_dict, test_dict