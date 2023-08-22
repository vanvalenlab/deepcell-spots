# Copyright 2019-2023 The Van Valen Lab at the California Institute of
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

"""Singleplex and multiplex FISH analysis application"""

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
from deepcell_spots.utils.preprocessing_utils import min_max_normalize
from deepcell_spots.utils.postprocessing_utils import max_cp_array_to_point_list_max
from deepcell_spots.multiplex import extract_spots_prob_from_coords_maxpool


def output_to_df(spots_locations_vec, cell_id_list, decoding_result):
    """
    Formats model output from lists and arrays to dataframe.

    Args:
        spots_locations_vec (numpy.array): An array of spots coordinates with
            shape `[num_spots, 3]`. The first two columns contain the coordinate
            locations of the spots and the last column contains the batch id of each
            spot.
        cell_id_list (numpy.array): An array of assigned cell id for each spot
            with shape `[num_spots,]`.
        decoding_result (dict): Keys include: `'probability'`, `'predicted_id'`,
            `'predicted_name'`.

    Returns:
        pandas.DataFrame: A dataframe combines all input information.
    """
    df = pd.DataFrame()
    df[['x', 'y', 'batch_id']] = spots_locations_vec.astype(np.int32)
    df['cell_id'] = cell_id_list
    for name, val in decoding_result.items():
        df[name] = val
    return df


class Polaris:
    """Loads spot detection and cell segmentation applications
    from deepcell_spots and deepcell_tf, respectively.

    The ``predict`` method calls the predict method of each
    application.

    Example::

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
        app = Polaris(image_type='singleplex')
        df_spots, df_intensities, segmentation_result = app.predict(
                             spots_image=spots_im,
                             segmentation_image=cyto_im)

        ####################################################################
        # Multiplex case:
        rounds = 10
        channels = 2
        df_barcodes = pd.read_csv('barcodes.csv', index_col=0)
        app = Polaris(image_type='singleplex',
                      decoding_kwargs={'rounds': rounds,
                                       'channels': channels,
                                       'df_barcodes': df_barcodes})
        df_spots, df_intensities, segmentation_result = app.predict(
                             spots_image=spots_im,
                             segmentation_image=cyto_im)

    Args:
        image_type (str): The type of the image. Valid values are
            `['singleplex' and 'multiplex']`. Defaults to `'singleplex'`.
        segmentation_model (tf.keras.Model): The model to load.
            If `None`, a pre-trained model will be downloaded.
        segmentation_type (str): The prediction type
            for generating segmentation predictions. Valid values
            are `['cytoplasm', 'nucleus', 'mesmer', 'no segmentation']`.
            Defaults to `'cytoplasm'`.
        spots_model (tf.keras.Model): The model to load.
            If `None`, a pre-trained model will be downloaded.
        decoding_kwargs (dict): Keyword arguments to pass to the decoding method.
            df_barcodes, rounds, channels. Defaults to empty, no decoding is performed.
            df_barcodes (pandas.DataFrame): Codebook, the first column is gene names (`'Gene'`),
                the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
                For exmaple, for a (rounds=10, channels=2) codebook, it should look the following
                (see `notebooks/Multiplex FISH Analysis.ipynb` for examples)::

                    Index:
                        RangeIndex (starting from 1)
                    Columns:
                        Name: Gene, dtype: object
                        Name: r0c0, dtype: int64
                        Name: r0c1, dtype: int64
                        Name: r1c0, dtype: int64
                        Name: r1c1, dtype: int64
                        ...
                        Name: r9c0, dtype: int64
                        Name: r9c1, dtype: int64

    Raises:
        ValueError: `image_type` must be one of `['singleplex', 'multiplex']`.
        ValueError: `segmentation_type` must be one of 
            `['cytoplasm', 'nucleus', 'mesmer', 'no segmentation']`.
    """

    def __init__(self,
                 image_type='singleplex',
                 segmentation_model=None,
                 segmentation_type='cytoplasm',
                 spots_model=None,
                 decoding_kwargs=None):

        self.spots_app = SpotDetection(model=spots_model)
        # Disable postprocessing_fn to return the full images
        self.spots_app.postprocessing_fn = None

        valid_image_types = ['singleplex', 'multiplex']
        if image_type not in valid_image_types:
            raise ValueError('Invalid image type supplied: {}. '
                             'Must be one of {}'.format(image_type,
                                                        valid_image_types))

        self.image_type = image_type
        if self.image_type == 'singleplex':
            self.decoding_app = None
            self.rounds = 1
            self.channels = 1
        elif self.image_type == 'multiplex':
            if not decoding_kwargs:
                self.decoding_app = None
                warnings.warn('No spot decoding application instantiated.')
            else:
                self.decoding_app = SpotDecoding(**decoding_kwargs)
                self.df_barcodes = decoding_kwargs['df_barcodes']
                self.rounds = decoding_kwargs['rounds']
                self.channels = decoding_kwargs['channels']
                if 'distribution' in decoding_kwargs.keys():
                    self.distribution = decoding_kwargs['distribution']
                else:
                    self.distribution = 'Relaxed Bernoulli'

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

    def _validate_prediction_input(self,
                                   spots_image,
                                   segmentation_image,
                                   background_image,
                                   threshold,
                                   mask_threshold):
        """Validate shape and values of prediction input.

        Args:
            spots_image (numpy.array): Input image for spot detection with shape
                `[batch, x, y, channel]`. Channel dimension should equal `rounds x channels`.
            segmentation_image (numpy.array): Input image for cell segmentation with shape
                `[batch, x, y, channel]`. Channel dimension should have a value of 1.
                Defaults to None.
            background_image (numpy.array): Input image for masking bright background objects with
                shape `[batch, x, y, channel]`. Channel dimension should less than or equal to
                the number of imaging channels. Defaults to None.
            threshold (float): Probability threshold for a pixel to be
                considered as a spot.
            mask_threshold (float): Percentile of pixel values in background image used to
                create a mask for bright background objects.

        Raises:
            ValueError: The size of the channel dimension of `spots_image` must equal
                `rounds * channels` for multiplex predictions.
            ValueError: The size of `spots_image` and `segmentation_image` must be the same, 
                excluding the channel dimension.
            ValueError: The channel dimension of `segmentation_image` must be 1.
            ValueError: The size of `spots_image` and `background_image` must be the same, 
                excluding the channel dimension.
            ValueError: The channel dimension of `background_image` must be less than or equal to
                `channels`.
            ValueError: `threshold` must be between 0 and 1.
            ValueError: `mask_threshold` must be between 0 and 1.
        """
        if self.image_type=='multiplex' and spots_image.shape[-1] != self.rounds * self.channels:
            raise ValueError('Shape of channel dimension of spots_image should equal to '
                             '(rounds x channels), but input segmentation_image had shape {} '
                             '(b,x,y,c).'.format(spots_image.shape))
        
        if segmentation_image is not None:
            if spots_image.shape[:-1] != segmentation_image.shape[:-1]:
                raise ValueError('Batch, x, and y dimensions of spots_image '
                                 'and segmentation_image must be the same. spots_image '
                                 'has shape {} and segmentation_image has shape {}'
                                 ''.format(spots_image.shape, segmentation_image.shape))

        if background_image is not None:
            if spots_image.shape[:-1] != background_image.shape[:-1]:
                raise ValueError('Batch, x, and y dimensions of spots_image '
                                 'and background_image must be the same. spots_image '
                                 'has shape {} and segmentation_image has shape {}'
                                 ''.format(spots_image.shape, background_image.shape))
            
            if background_image.shape[-1] > self.channels:
                raise ValueError('Shape of channel dimension of background_image should be less '
                                 'than or equal to the number of imaging channels, but input '
                                 'segmentation_image had shape {} (b,x,y,c).'
                                 ''.format(background_image.shape))

        if threshold < 0 or threshold > 1:
            raise ValueError('Input threshold was %s. Threshold value must be '
                             'between 0 and 1.'.format())

        if mask_threshold < 0 or mask_threshold > 1:
            raise ValueError('Input mask_threshold was %s. Threshold value must be '
                             'between 0 and 1.'.format())

    def _predict_spots_image(self, spots_image, clip, skip_round):
        """Iterate through all channels and generate model output (pixel-wise spot probability).

        Args:
            spots_image (numpy.array): Input image for spot detection with shape
                ``[batch, x, y, channel]``.
            clip (bool): Determines if pixel values will be clipped by percentile.
                Defaults to `True`.
            skip_round(list): List of boolean values for whether an imaging round is used in the
                defined codebook.

        Returns:
            numpy.array: Output probability map with shape ``[batch, x, y, channel]``.
        """

        output_image = np.zeros_like(spots_image, dtype=np.float32)
        for idx_round in range(spots_image.shape[-1]):
            if skip_round[idx_round]:
                continue
            output_image[..., idx_round] = self.spots_app.predict(
                image=spots_image[..., idx_round:idx_round+1],
                clip=clip
            )['classification'][..., 1]
        return output_image

    def _mask_spots(self, spots_locations, background_image, mask_threshold):
        """Mask predicted spots in regions of high background intensity. If input background
        image contains more than one channel, background mask will be maximum intensity projected
        across channel axis.

        Args:
            spots_locations (list): A list of length `batch` containing arrays of spots
                coordinates with shape `[num_spots, 2]`.
            background_image (numpy.array): Input image for masking bright background objects with
                shape `[batch, x, y, channel]`.
            mask_threshold (float): Percentile of pixel values in background image used to
                create a mask for bright background objects.

        Returns:
            array: Array with values 0 and 1, whether predicted spot is within a masked backround
                object.
        """
        normalized_image = np.zeros(background_image.shape)
        for i in range(background_image.shape[0]):
            normalized_image[i] = min_max_normalize(background_image[i:i+1], clip=True)
        mask = normalized_image > mask_threshold
        mask = np.max(mask, axis=-1)

        result = np.zeros(np.vstack(spots_locations).shape[0])
        for i in range(len(spots_locations)):
            for ii, spot in enumerate(spots_locations[i]):
                if mask[i, int(spot[0]), int(spot[1])] == 1:
                    if i == 0:
                        spot_index = ii
                    else:
                        spot_index = np.vstack(spots_locations[:i]).shape[0] + ii
                    result[spot_index] = 1

        return(result)

    def _predict(self,
                 spots_image,
                 segmentation_image,
                 background_image,
                 image_mpp,
                 threshold,
                 clip,
                 mask_threshold,
                 maxpool_extra_pixel_num,
                 decoding_training_kwargs):
        """Generates prediction output consisting of a labeled cell segmentation image,
        detected spot locations, and a dictionary of spot locations assigned to labeled
        cells of the input.

        Input images are required to have 4 dimensions `[batch, x, y, channel]`. Additional
        empty dimensions can be added using ``np.expand_dims``.

        Args:
            spots_image (numpy.array): Input image for spot detection with shape
                `[batch, x, y, channel]`. Channel dimension should equal `rounds x channels`.
            segmentation_image (numpy.array): Input image for cell segmentation with shape
                `[batch, x, y, channel]`. Channel dimension should have a value of 1.
                Defaults to None.
            background_image (numpy.array): Input image for masking bright background objects with
                shape `[batch, x, y, channel]`. Channel dimension should less than or equal to
                the number of imaging channels. Defaults to None.
            image_mpp (float): Microns per pixel for `spots_image`.
            threshold (float): Probability threshold for a pixel to be decoded.
            clip (bool): Determines if pixel values will be clipped by percentile.
                Defaults to `True`.
            mask_threshold (float): Percentile of pixel values in background image used to
                create a mask for bright background objects.
            maxpool_extra_pixel_num (int): Number of extra pixel for max pooling. Defaults
                to 0, means no max pooling. For any number t, there will be a pool with
                shape `[-t, t] x [-t, t]`.
            decoding_training_kwargs (dict): Including `num_iter`, `batch_size`,
                `pred_prob_thresh`.
        Raises:
            ValueError: The size of the channel dimension of `spots_image` must equal
                `rounds * channels` for multiplex predictions.
            ValueError: The size of `spots_image` and `segmentation_image` must be the same, 
                excluding the channel dimension.
            ValueError: The channel dimension of `segmentation_image` must be 1.
            ValueError: The size of `spots_image` and `background_image` must be the same, 
                excluding the channel dimension.
            ValueError: The channel dimension of `background_image` must be less than or equal to
                `channels`.
            ValueError: `threshold` must be between 0 and 1.
            ValueError: `mask_threshold` must be between 0 and 1.
            ValueError: Segmentation application must be instantiated if segmentation
                image is defined.

        Returns:
            df_spots (pandas.DataFrame): Columns are `x`, `y`, `batch_id`, `cell_id`,
                `probability`, `predicted_id`, `predicted_name`, `spot_index`, and `source`.
                `cell_id = 0` means the spot is outside the cells or tissue. Rows with the same
                `spot_index` resulted from a spot with two mixed barcodes. The values of `source`
                can include `prediction` (result of `SpotDecoding`), `error rescue` (spots rescued
                based on their Hamming distance to a gene in the code book), and `mixed rescue`
                (spots rescued from a spot with two mixed barcodes).
            df_intensities (pandas.DataFrame): Columns are channels and rows are spots.
            segmentation_result (numpy.array): Segmentation mask with shape `[batch, x, y, 1]`.
        """
        self._validate_prediction_input(spots_image, segmentation_image, background_image,
                                        threshold, mask_threshold)
        if self.image_type == 'multiplex':
            skip_round = np.array(np.sum(self.df_barcodes.iloc[:,1:], axis=0)==0)
        else:
            skip_round = [False]*np.shape(spots_image)[-1]
        output_image = self._predict_spots_image(spots_image, clip, skip_round)

        clipped_output_image = np.clip(output_image, 0, 1)
        max_proj_images = np.max(clipped_output_image, axis=-1)
        spots_locations = [np.argwhere(max_proj_images[i]>threshold) for i in range(max_proj_images.shape[0])]

        if self.image_type == 'multiplex':
            if self.distribution == 'Gaussian':
                norm_spots_image = min_max_normalize(spots_image)
                spots_intensities = extract_spots_prob_from_coords_maxpool(
                    norm_spots_image, spots_locations, extra_pixel_num=maxpool_extra_pixel_num)
            elif self.distribution == 'Relaxed Bernoulli':
                spots_intensities = extract_spots_prob_from_coords_maxpool(
                    clipped_output_image, spots_locations, extra_pixel_num=maxpool_extra_pixel_num)
            elif self.distribution == 'Bernoulli':
                spots_intensities = extract_spots_prob_from_coords_maxpool(
                    clipped_output_image, spots_locations, extra_pixel_num=maxpool_extra_pixel_num)
                spots_intensities = np.array(np.array(spots_intensities) > 0.5).astype('int')
        else:
            spots_intensities = extract_spots_prob_from_coords_maxpool(
                    spots_image, spots_locations, extra_pixel_num=maxpool_extra_pixel_num)
        spots_intensities_vec = np.concatenate(spots_intensities)
        spots_locations_vec = np.concatenate([np.concatenate(
            [item, [[idx_batch]] * len(item)], axis=1)
            for idx_batch, item in enumerate(spots_locations)])

        if segmentation_image is not None:
            if not self.segmentation_app:
                raise ValueError('Segmentation application must be instantiated if '
                                 'segmentation image is defined.')
            else:
                segmentation_result = self.segmentation_app.predict(segmentation_image,
                                                                    image_mpp=image_mpp)
                spots_cell_assignments_vec = match_spots_to_cells_as_vec_batched(
                    segmentation_result, spots_locations)
        else:
            segmentation_result = None
            spots_cell_assignments_vec = None

        if self.decoding_app is not None:
            decoding_result = self.decoding_app.predict(
                spots_intensities_vec, **decoding_training_kwargs)
            spots_locations_ext = spots_locations_vec[decoding_result['spot_index']]
        else:
            decoding_result = {'spot_index': None,
                               'probability': None,
                               'predicted_id': None,
                               'predicted_name': None,
                               'source': None}
            spots_locations_ext = spots_locations_vec

        if background_image is not None:
            decoding_result['masked'] = self._mask_spots(spots_locations,
                                                         background_image,
                                                         mask_threshold)

        df_spots = output_to_df(spots_locations_ext,
                                spots_cell_assignments_vec,
                                decoding_result)
        df_intensities = pd.DataFrame(spots_intensities_vec)
        df_results = pd.concat([df_spots, df_intensities], axis=1)

        if self.image_type == 'multiplex':
            dec_prob_im = np.zeros((spots_image.shape[:3]))

            for i in range(len(df_results)):
                gene = df_results.loc[i, 'predicted_name']
                if gene in ['Background', 'Unknown']:
                    continue
                if 'Blank' in gene:
                    continue
                
                x = df_results.loc[i, 'x']
                y = df_results.loc[i, 'y']
                b = df_results.loc[i, 'batch_id']
                prob = max_proj_images[b, x, y]
                
                dec_prob_im[b, x, y] = prob

            decoded_spots_locations = max_cp_array_to_point_list_max(dec_prob_im,
                                                                    threshold=None, min_distance=1)
            mask = []
            for i in range(np.shape(decoded_spots_locations)[1]):
                x = decoded_spots_locations[0][i, 0]
                y = decoded_spots_locations[0][i, 1]

                mask.append(df_results.loc[(df_results.x==x) & (df_results.y==y)].index[0])
                
            df_results = df_results.loc[mask]

        return df_results, segmentation_result

    def predict(self,
                spots_image,
                segmentation_image=None,
                background_image=None,
                image_mpp=None,
                threshold=0.01,
                clip=True,
                mask_threshold=0.5,
                maxpool_extra_pixel_num=0,
                decoding_training_kwargs={}):
        """Generates prediction output consisting of a labeled cell segmentation image,
        detected spot locations, and a dictionary of spot locations assigned to labeled
        cells of the input.

        Input images are required to have 4 dimensions `[batch, x, y, channel]`. Additional
        empty dimensions can be added using ``np.expand_dims``.

        Args:
            spots_image (numpy.array): Input image for spot detection with shape
                `[batch, x, y, channel]`. Channel dimension should equal `rounds x channels`.
            segmentation_image (numpy.array): Input image for cell segmentation with shape
                `[batch, x, y, channel]`. Channel dimension should have a value of 1.
                Defaults to None.
            background_image (numpy.array): Input image for masking bright background objects with
                shape `[batch, x, y, channel]`. Channel dimension should less than or equal to
                the number of imaging channels. Defaults to None.
            image_mpp (float): Microns per pixel for `spots_image`.
            threshold (float): Probability threshold for a pixel to be
                considered as a spot.
            clip (bool): Determines if pixel values will be clipped by percentile.
                Defaults to `True`.
            mask_threshold (float): Percentile of pixel values in background image used to
                create a mask for bright background objects.
            maxpool_extra_pixel_num (int): Number of extra pixel for max pooling. Defaults
                to 0, means no max pooling. For any number t, there will be a pool with
                shape `[-t, t] x [-t, t]`.
            decoding_training_kwargs (dict): Including `num_iter`, `batch_size`,
                `pred_prob_thresh`.
        Raises:
            ValueError: The size of the channel dimension of `spots_image` must equal
                `rounds * channels` for multiplex predictions.
            ValueError: The size of `spots_image` and `segmentation_image` must be the same, 
                excluding the channel dimension.
            ValueError: The channel dimension of `segmentation_image` must be 1.
            ValueError: The size of `spots_image` and `background_image` must be the same, 
                excluding the channel dimension.
            ValueError: The channel dimension of `background_image` must be less than or equal to
                `channels`.
            ValueError: `threshold` must be between 0 and 1.
            ValueError: `mask_threshold` must be between 0 and 1.
            ValueError: Segmentation application must be instantiated if segmentation
                image is defined.

        Returns:
            df_spots (pandas.DataFrame): Columns are `x`, `y`, `batch_id`, `cell_id`,
                `probability`, `predicted_id`, `predicted_name`. `cell_id = 0` means
                the spot is outside the cells or tissue.
            df_intensities (pandas.DataFrame): Columns are channels and rows are spots.
            segmentation_result (numpy.array): Segmentation mask with shape `[batch, x, y, 1]`.
        """

        return self._predict(
            spots_image=spots_image,
            segmentation_image=segmentation_image,
            background_image=background_image,
            image_mpp=image_mpp,
            threshold=threshold,
            clip=clip,
            mask_threshold=mask_threshold,
            maxpool_extra_pixel_num=maxpool_extra_pixel_num,
            decoding_training_kwargs=decoding_training_kwargs
        )
