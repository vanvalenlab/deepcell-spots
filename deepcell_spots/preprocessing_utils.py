import numpy as np 
import logging

def mean_std_normalize(image, epsilon=1e-07):
    """Normalize image data by dividing by the maximum pixel value
    Args:
        image (numpy.array): numpy array of image data
        epsilon (float): fuzz factor used in numeric expressions.
    Returns:
        numpy.array: normalized image data
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = (img - img.mean()) / (img.std() + epsilon)
            image[batch, ..., channel] = normal_image
    return image

def min_max_normalize(image):
    """Normalize image data by dividing by the maximum pixel value
    Args:
        image (numpy.array): numpy array of image data
        epsilon (float): fuzz factor used in numeric expressions.
    Returns:
        numpy.array: normalized image data
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            min_val = min(map(min,img))
            max_val = max(map(max,img))
            normal_image = (img - min_val) / (max_val - min_val)
            image[batch, ..., channel] = normal_image
    return image