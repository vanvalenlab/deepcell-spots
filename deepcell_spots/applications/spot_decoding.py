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

"""Spot decoding application"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from scipy.spatial import distance
from tqdm import tqdm

from deepcell.applications import Application
from deepcell_spots.decoding_functions import decoding_function


class SpotDecoding(Application):
    """Initialize a model for spot decoding of multiplex images.

    The ``predict`` method handles inference procedure. It infers the
    model parameters and predicts the spot identities.

    Example::

        from deepcell_spots.applications import SpotDecoding

        # Create the application
        app = SpotDecoding(df_barcodes, rounds, channels)

        # Decode the spots
        decoding_dict = app.predict(spots_intensities_vec)

    Args:
        df_barcodes (pandas.DataFrame): Codebook, the first column is gene names ('Gene'),
            the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
            For exmaple, for a (rounds=10, channels=2) codebook, it should look like::
            
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

        rounds (int): Number of rounds.
        channels (int): Number of channels.
        distribution (str): Distribution for spot intensities in spot decoding model. Valid options:
            ['Gaussian', 'Bernoulli', 'Relaxed Bernoulli']. Defaults to 'Relaxed Bernoulli'.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Bernoulli or Relaxed Bernoulli distributions.
            Valid options: ['2', '2*R', '2*C', '2*R*C', 'Gaussian']. Defaults to '2*R*C'. 
    """

    dataset_metadata = {}
    model_metadata = {}

    def __init__(self,
                 df_barcodes,
                 rounds,
                 channels,
                 distribution='Relaxed Bernoulli',
                 params_mode='2*R*C'):
        self.rounds = rounds
        self.channels = channels
        self.distribution = distribution
        self.params_mode = params_mode

        self._validate_codebook(df_barcodes)
        self.df_barcodes = self._add_bkg_unknown_to_barcodes(df_barcodes)
        

        super(SpotDecoding, self).__init__(
            model=None,
            model_image_shape=[0],
            model_mpp=None,
            preprocessing_fn=None,
            postprocessing_fn=None,
            format_model_output_fn=None,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)

    def _validate_codebook(self, df_barcodes):
        """Validate the format of the input codebook.

        Args:
            df_barcodes (pandas.DataFrame): Codebook, the first column is gene names ('Gene'),
            the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
            For exmaple, for a (rounds=10, channels=2) codebook, it should look like::
            
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
        """
        if not isinstance(df_barcodes, pd.DataFrame):
            raise TypeError('df_barcodes must be a Pandas DataFrame')
        
        if df_barcodes.columns[0] != 'Gene':
            raise ValueError('The first column of df_barcodes must contain the gene names and '
                             'have the column name \'Gene\'.')
        
        if len(df_barcodes.columns) != self.rounds * self.channels + 1:
            raise ValueError('The length of the barcode must equal rounds*channels.')
        
        valid_vals = [0,1]
        vals = df_barcodes.values[:, 1:]
        if set(valid_vals) != set(vals.flatten()):
            raise ValueError('Barcode values must be 0 or 1.')

        if 'Background' in df_barcodes.columns or 'Unknown' in df_barcodes.columns:
            raise ValueError('Codebook should not include \'Background\' or \'Unknown\' values. '
                             'These values will be added automatically.')

    def _add_bkg_unknown_to_barcodes(self, df_barcodes):
        """Add Background and Unknown category to the codebook. The barcode of Background
        is all zeros and the barcode for Unknown is all -1s.

        Args:
            df_barcodes (pd.DataFrame): The codebook initialized by users.

        Returns:
            pd.DataFrame: The augmented codebook.

        """
        df_barcodes_aug = df_barcodes.copy()
        barcode_len = df_barcodes_aug.shape[1] - 1
        df_barcodes_aug.loc[len(df_barcodes_aug)+1] = ['Background'] + [0] * (barcode_len)
        df_barcodes_aug.loc[len(df_barcodes_aug)+1] = ['Unknown'] + [-1] * (barcode_len)
        return df_barcodes_aug
    
    def _validate_spots_intensities(self, spots_intensities_vec):
        """Validate values of spot intensities before spot decoding prediction.

        Args:
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                [num_spots, r*c].
        """
        if self.distribution == 'Relaxed Bernoulli':
            if (spots_intensities_vec > 1).any() or (spots_intensities_vec < 0).any():
                raise ValueError('Spot intensities should be between 0 and 1 when '
                                 'distribution=\'Relaxed Bernoulli\'.')
            
            if self.distribution == 'Bernoulli':
                if set([0,1]) != set(spots_intensities_vec.flatten()):
                    raise ValueError('Spot intensities should be 0 or 1 when '
                                 'distribution=\'Bernoulli\'.')

    def _decoding_output_to_dict(self, out):
        """Convert decoding output to dictionary.

        Args:
            out (dict): Dictionary with keys: 'class_probs', 'params'.

        Returns:
            dict: Dictionary with keys: 'probability', 'predicted_id', 'predicted_name'.

        """
        barcodes_idx2name = dict(
            zip(1 + np.arange(len(self.df_barcodes)), self.df_barcodes.Gene.values))
        decoded_dict = {}
        decoded_dict['probability'] = out['class_probs'].max(axis=1)
        decoded_dict['predicted_id'] = out['class_probs'].argmax(axis=1) + 1
        decoded_dict['predicted_name'] = np.array(
            list(map(barcodes_idx2name.get, decoded_dict['predicted_id'])))
        return decoded_dict

    def _threshold_unknown_by_prob(self, decoded_dict, unknown_index, thres_prob=0.5):
        """Threshold the decoded spots to identify unknown. If the highest probability
        if below a certain threshold, the spot will be classfied as Unknown.

        Args:
            decoded_dict (dict): Dictionary containing decoded spot identities with
                keys: 'probability', 'predicted_id', 'predicted_name'.
            unknown_index (int): The index for Unknown category.

        Returns:
            dict: similar to input, just replace the low probability
                ones with Unknown.
        """
        decoded_dict['predicted_id'][decoded_dict['probability'] < thres_prob] = unknown_index
        decoded_dict['predicted_name'][decoded_dict['probability'] < thres_prob] = 'Unknown'
        return decoded_dict

    def _rescue_spots(self,
                      decoding_dict_trunc,
                      spots_intensities_vec):
        """Rescues decoded spots assigned as 'Background' or 'Unknown' by if their spot
        probability values have a Hamming distance of 1 from each of the barcodes.

        Args:
            decoding_dict_trunc (dict): Dictionary containing decoded spot identities with
                keys: 'probability', 'predicted_id', 'predicted_name'. This dictionary has already
                been processed to assign low probability predictions to 'Unknown'.
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                [num_spots, r*c].
        """

        ch_names = list(self.df_barcodes.columns)
        ch_names.remove('Gene')
        barcodes_array = self.df_barcodes[ch_names].values
        num_barcodes = barcodes_array.shape[0]
        barcode_len = barcodes_array.shape[1]

        predicted_ids = decoding_dict_trunc['predicted_id']
        predicted_names = decoding_dict_trunc['predicted_name']

        for i,pred in tqdm(enumerate(predicted_names)):
            if pred in ['Background', 'Unknown']:
                dist_list = np.zeros(num_barcodes)
                for ii in range(num_barcodes):
                    dist_list[ii] = distance.hamming(np.round(spots_intensities_vec[i]),
                                                     barcodes_array[ii])
                scaled_dist_list = dist_list * barcode_len
                if 1 in scaled_dist_list:
                    new_gene = np.argwhere(scaled_dist_list == 1)[0][0]
                    predicted_ids[i] = new_gene
                    predicted_names[i] = self.df_barcodes['Gene'].values[new_gene]
        
        decoding_dict_rescued = {
            'predicted_id': predicted_ids,
            'predicted_name': predicted_names,
            'probability': decoding_dict_trunc['probability']
        }
        
        return(decoding_dict_rescued)

    def _predict(self,
                 spots_intensities_vec,
                 num_iter,
                 batch_size,
                 thres_prob,
                 rescue_spots):
        """Predict the gene assignment of each spot.

        Args:
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                [num_spots, r*c].
            num_iter (int): Number of iterations for training. Defaults to 500.
            batch_size (int): Size of batches for training. Defaults to 1000.
            thres_prob (float): The threshold of unknown category, within [0,1]. Defaults to 0.5.
            rescue_spots (bool): Whether to check if 'Background'-  and 'Unknown'-assigned spots
                have a Hamming distance of 1 to other barcodes.

        Returns:
            dict: Dictionary with keys: 'probability', 'predicted_id', 'predicted_name'.
        """
        self._validate_spots_intensities(spots_intensities_vec)

        spots_intensities_reshaped = np.reshape(spots_intensities_vec,
                                                (-1, self.channels, self.rounds))

        # convert df_barcodes to an array
        ch_names = list(self.df_barcodes.columns)
        ch_names.remove('Gene')
        unknown_index = self.df_barcodes.index.max()
        barcodes_array = self.df_barcodes[ch_names].values.reshape(-1, self.channels,
                                                                   self.rounds)[:-1, :, :]

        # decode
        out = decoding_function(spots_intensities_reshaped,
                                barcodes_array,
                                num_iter=num_iter,
                                batch_size=batch_size,
                                distribution=self.distribution,
                                params_mode=self.params_mode)
        decoding_dict = self._decoding_output_to_dict(out)
        decoding_dict_trunc = self._threshold_unknown_by_prob(
            decoding_dict, unknown_index, thres_prob=thres_prob)

        if rescue_spots:
            decoding_dict_rescued = self._rescue_spots(decoding_dict_trunc,
                                                       spots_intensities_vec)
            return decoding_dict_rescued
        else:
            return decoding_dict_trunc

    def predict(self,
                spots_intensities_vec,
                num_iter=500,
                batch_size=1000,
                thres_prob=0.5,
                rescue_spots=True):
        """Predict the gene assignment of each spot.

        Args:
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                [num_spots, r*c].
            num_iter (int): Number of iterations for training. Defaults to 500.
            batch_size (int): Size of batches for training. Defaults to 1000.
            thres_prob (float): The threshold of unknown category, within [0,1]. Defaults to 0.5.
            rescue_spots (bool): Whether to check if 'Background' and 'Unknown' assigned spots
                have a Hamming distance of 1 to other barcodes.

        Returns:
            dict: Dictionary with keys: 'probability', 'predicted_id', 'predicted_name'.
        """

        return self._predict(
            spots_intensities_vec=spots_intensities_vec,
            num_iter=num_iter,
            batch_size=batch_size,
            thres_prob=thres_prob,
            rescue_spots=rescue_spots)
