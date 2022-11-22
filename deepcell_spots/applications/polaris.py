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
"""Singleplex FISH analysis application"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np
import pandas as pd

from deepcell.applications import CytoplasmSegmentation, NuclearSegmentation
from deepcell.applications import Mesmer
from deepcell_spots.applications import SpotDetection, SpotDecoding
from deepcell_spots.singleplex import match_spots_to_cells_as_vec_batched
from deepcell_toolbox.processing import histogram_normalization
from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell_spots.postprocessing_utils import max_cp_array_to_point_list_max
from deepcell_spots.multiplex import extract_spots_prob_from_coords_maxpool


def output_to_df(spots_locations_vec, cell_id_list, decoding_result):
    """
    Formats model output from lists and arrays to dataframe.

    Args:
        spots_locations_vec (numpy.array): An array of spots coordinates with 
            shape ``[num_spots, 2]``.
        cell_id_list (numpy.array): An array of assigned cell id for each spot
            with shape ``[num_spots,]``.
        decoding_result (dict): Keys include: 'probability', 'predicted_id', 
            'predicted_name'. 

    Returns:
        pandas.DataFrame: A dataframe combines all input information.
    """
    df = pd.DataFrame()
    df[['x', 'y', 'batch_id']] = spots_locations_vec.astype(np.int32)
    df['cell_id'] = cell_id_list
    for name, val in decoding_result.items():
        df[name] = val
    return df


class Polaris(object):
    """Loads spot detection and cell segmentation applications
    from deepcell_spots and deepcell_tf, respectively.

    The ``predict`` method calls the predict method of each
    application.

    Example:

    .. code-block:: python

        from skimage.io import imread
        from deepcell_spots.applications import Polaris

        # Load the images
        spots_im = imread('spots_image.png')
        cyto_im = imread('cyto_image.png')

        # Expand image dimensions to rank 4
        spots_im = np.expand_dims(spots_im, axis=[0,-1])
        cyto_im = np.expand_dims(cyto_im, axis=[0,-1])
        
        ####################################################################
        # Singleplex case:
        app = Polaris(image_type='singplex')
        df_spots, df_intensities, segmentation_result = app.predict(
                             spots_image=spots_im,
                             segmentation_image=cyto_im)
        
        ####################################################################
        # Multiplex case:
        r = 10
        c = 2
        df_barcodes = pd.read_csv('barcodes.csv', index_col=0)
        app = Polaris(image_type='singplex', 
                      decoding_kwargs={'r': r, 
                                       'c': c, 
                                       'df_barcodes': df_barcodes})
        df_spots, df_intensities, segmentation_result = app.predict(
                             spots_image=spots_im,
                             segmentation_image=cyto_im)

    Args:
        image_type (str): The type of the image. Valid values are 
            'singleplex' and 'multiplex'. Defaults to 'singleplex'.
        segmentation_model (tf.keras.Model): The model to load.
            If ``None``, a pre-trained model will be downloaded.
        segmentation_compartment (str): The cellular compartment
            for generating segmentation predictions. Valid values
            are 'cytoplasm', 'nucleus', 'no segmentation'. Defaults
            to 'cytoplasm'.
        spots_model (tf.keras.Model): The model to load.
            If ``None``, a pre-trained model will be downloaded.
        decoding_kwargs (dict): Keyword arguments to pass to the decoding method.
            df_barcodes, r, c. Defaults to empty, no decoding is performed.
            df_barcodes (pandas.DataFrame): Codebook, one column is gene names ('code_name'),
                the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
                For exmaple, for a (r=10, c=2) codebook, it should look like:
                    Index:
                        RangeIndex (starting from 1)
                    Columns:
                        Name: code_name, dtype: object
                        Name: r0c0, dtype: int64
                        Name: r0c1, dtype: int64
                        Name: r1c0, dtype: int64
                        Name: r1c1, dtype: int64
                        ...
                        Name: r9c0, dtype: int64
                        Name: r9c1, dtype: int64
    """

    def __init__(self,
                 image_type='singplex',
                 segmentation_model=None,
                 segmentation_type='cytoplasm',
                 spots_model=None,
                 decoding_kwargs={}):

        self.spots_app = SpotDetection(model=spots_model)
        self.spots_app.postprocessing_fn = None # Disable postprocessing_fn to return the full images
        
        valid_image_types = ['singplex', 'multiplex']
        if image_type not in valid_image_types:
            raise ValueError('Invalid image type supplied: {}. '
                             'Must be one of {}'.format(image_type,
                                                        valid_image_types))

        self.image_type = image_type
        if self.image_type == 'singplex':
            self.decoding_app = None
        elif self.image_type == 'multiplex':
            if not decoding_kwargs:
                self.decoding_app = None
                warnings.warn('No spot decoding application instantiated.')
            else:
                self.decoding_app = SpotDecoding(**decoding_kwargs)

        valid_compartments = ['cytoplasm', 'nucleus', 'mesmer', 'no segmentation']
        if segmentation_type not in valid_compartments:
            raise ValueError('Invalid compartment supplied: {}. '
                             'Must be one of {}'.format(segmentation_type,
                                                        valid_compartments))

        if segmentation_type == 'cytoplasm':
            self.segmentation_app = CytoplasmSegmentation(model=segmentation_model)
            self.segmentation_app.preprocessing_fn = histogram_normalization
            self.segmentation_app.postprocessing_fn = deep_watershed
        elif segmentation_type == 'nucleus':
            self.segmentation_app = NuclearSegmentation(model=segmentation_model)
        elif segmentation_type == 'mesmer':
            self.segmentation_app = Mesmer()
        else:
            self.segmentation_app = None
            warnings.warn('No segmentation application instantiated.')

    def _predict_spots_image(self, spots_image, spots_threshold, spots_clip):
        """Iterate through all channels and generate model output (probability maps)

        Args:
            spots_image (numpy.array): Input image for spot detection with shape
                ``[batch, x, y, channel]``.
            spots_threshold (float): Probability threshold for a pixel to be
                considered as a spot.
            spots_clip (bool): Determines if pixel values will be clipped by percentile.
                Defaults to false.

        Returns:
            numpy.array: Output probability map with shape ``[batch, x, y, channel]``.
        """
        
        output_image = np.zeros_like(spots_image, dtype=np.float32)
        for idx_channel in range(spots_image.shape[-1]):
            output_image[..., idx_channel] = self.spots_app.predict(
                image=spots_image[..., idx_channel][..., None],
                threshold=spots_threshold, #TODO: threshold is disabled, but must feed a float [0,1] number
                clip=spots_clip
            )['classification'][...,1]
        return output_image

    def predict(self,
                spots_image,
                segmentation_image=None,
                image_mpp=None,
                spots_threshold=0.95,
                spots_clip=False,
                maxpool_extra_pixel_num=1,
                decoding_training_kwargs={}):
        """Generates prediction output consisting of a labeled cell segmentation image,
        detected spot locations, and a dictionary of spot locations assigned to labeled
        cells of the input.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``. Channel dimension should be 1.

        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            spots_image (numpy.array): Input image for spot detection with shape
                ``[batch, x, y, channel]``.
            segmentation_image (numpy.array): Input image for cell segmentation with shape
                ``[batch, x, y, channel]``. Defaults to None.
            image_mpp (float): Microns per pixel for ``image``.
            spots_threshold (float): Probability threshold for a pixel to be
                considered as a spot.
            spots_clip (bool): Determines if pixel values will be clipped by percentile.
                Defaults to false.
            maxpool_extra_pixel_num (int): Number of extra pixel for max pooling. Defaults to 1, 
                meaning a pool with size = 9: (1,0,-1)x(1,0,-1). 0 means no maxpooling.
            decoding_training_kwargs (dict): Including num_iter, batch_size, thres_prob.
        Raises:
            ValueError: Threshold value must be between 0 and 1.
            ValueError: Segmentation application must be instantiated if segmentation
                image is defined.

        Returns:
            df_spots (pandas.DataFrame): Columns are x, y, batch_id, cell_id, probability, predicted_id, preicted_name.
            df_intensities (pandas.DataFrame): Columns are channels and rows are spots.
            segmentation_result (numpy.array): Segmentation mask with shape ``[batch, x, y, 1]``.
        """
        if spots_threshold < 0 or spots_threshold > 1:
            raise ValueError('Threshold of %s was input. Threshold value must be '
                             'between 0 and 1.'.format())

        output_image = self._predict_spots_image(spots_image, spots_threshold, spots_clip)
    
        max_proj_images = np.max(output_image, axis=-1)
        spots_locations = max_cp_array_to_point_list_max(max_proj_images, 
            threshold=spots_threshold, min_distance=1)
        
        spots_intensities = extract_spots_prob_from_coords_maxpool(output_image, spots_locations, extra_pixel_num=maxpool_extra_pixel_num)
        spots_intensities_vec = np.concatenate(spots_intensities)
        spots_locations_vec = np.concatenate([np.concatenate([item, [[idx_batch]]*len(item)], axis=1) \
            for idx_batch, item in enumerate(spots_locations)])

        if segmentation_image is not None:
            if not self.segmentation_app:
                raise ValueError('Segmentation application must be instantiated if '
                                 'segmentation image is defined.')
            else:
                segmentation_result = self.segmentation_app.predict(segmentation_image,
                                                                    image_mpp=image_mpp)
                spots_cell_assignments_vec = match_spots_to_cells_as_vec_batched(segmentation_result, spots_locations)
        else:
            segmentation_result = None
            spots_cell_assignments_vec = None
            
        if self.decoding_app is not None:
            decoding_result = self.decoding_app.predict(spots_intensities_vec, **decoding_training_kwargs)
        else:
            decoding_result = {'probability': None, 'predicted_id': None, 'predicted_name': None}

        df_spots = output_to_df(spots_locations_vec, spots_cell_assignments_vec, decoding_result)
        df_intensities = pd.DataFrame(spots_intensities_vec)
        return df_spots, df_intensities, segmentation_result

