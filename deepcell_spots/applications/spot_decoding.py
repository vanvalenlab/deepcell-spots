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


class SpotDecoding:
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
        df_barcodes (pandas.DataFrame): Codebook, the first column is gene names (`'Gene'`),
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
            `['Gaussian', 'Bernoulli', 'Relaxed Bernoulli']`. Defaults to `'Relaxed Bernoulli'`.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Bernoulli or Relaxed Bernoulli distributions.
            Valid options: `['2', '2*R', '2*C', '2*R*C', 'Gaussian']`. Defaults to `'2*R*C'`. 

    Raises:
        ValueError: `df_barcodes` must be a Pandas DataFrame.
        ValueError: The first column of `df_barcodes` must be called `'Gene'`. It should contain
            the names of the genes in the codebook.
        ValueError: The number of columns in `df_barcodes` must equal `rounds * channels + 1`.
        ValueError: Barcode values must be 0 or 1.
        ValueError: Codebook genes should not include `'Background'` or `'Unknown'`.
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

    def _validate_codebook(self, df_barcodes):
        """Validate the format of the input codebook.

        Args:
            df_barcodes (pandas.DataFrame): Codebook, the first column is gene names (`'Gene'`),
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
            raise ValueError('The length of the barcode must equal rounds * channels.')
        
        valid_vals = {0,1}
        vals = df_barcodes.values[:, 1:]
        if valid_vals != set(vals.flatten()):
            raise ValueError('Barcode values must be 0 or 1.')

        if 'Background' in df_barcodes.columns or 'Unknown' in df_barcodes.columns:
            raise ValueError('Codebook should not include \'Background\' or \'Unknown\' values. '
                             'These values will be added automatically.')

    def _add_bkg_unknown_to_barcodes(self, df_barcodes):
        """Add `Background` and `Unknown` barcodes to the codebook. The barcode of
        `Background` is all zeros and the barcode for `Unknown` is all -1s.

        Args:
            df_barcodes (pd.DataFrame): Codebook, the first column is gene names (`'Gene'`),
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
                [num_spots, (rounds * channels)].

        Raises:
            ValueError: Spot intensities should be bewteen 0 and 1 when `distribution` is
                `'Relaxed Bernoulli'`.
            ValueError: Spot intensities should be 0 or 1 when `distribution` is `'Bernoulli'`.
        """
        if self.distribution == 'Relaxed Bernoulli':
            if (spots_intensities_vec > 1).any() or (spots_intensities_vec < 0).any():
                raise ValueError('Spot intensities should be between 0 and 1 when '
                                 'distribution=\'Relaxed Bernoulli\'.')
            
        if self.distribution == 'Bernoulli':
            if {0,1} != set(spots_intensities_vec.flatten()):
                raise ValueError('Spot intensities should be 0 or 1 when '
                                    'distribution=\'Bernoulli\'.')

    def _decoding_output_to_dict(self, out):
        """Convert decoding output to dictionary.

        Args:
            out (dict): Dictionary with keys: `'class_probs'`, `'params'`.

        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.

        """
        barcodes_idx2name = dict(
            zip(1 + np.arange(len(self.df_barcodes)), self.df_barcodes.Gene.values))
        decoded_dict = {}
        decoded_dict['probability'] = out['class_probs'].max(axis=1)
        decoded_dict['spot_index'] = np.arange(len(decoded_dict['probability']))
        decoded_dict['predicted_id'] = out['class_probs'].argmax(axis=1) + 1
        decoded_dict['predicted_name'] = np.array(
            list(map(barcodes_idx2name.get, decoded_dict['predicted_id'])))
        decoded_dict['source'] = np.repeat(
            'prediction', len(decoded_dict['probability'])).astype('U25')
        return decoded_dict

    def _threshold_unknown_by_prob(self, decoded_dict, unknown_index, pred_prob_thresh=0.95):
        """Threshold the decoded spots to identify unknown. If the highest probability
        if below a certain threshold, the spot will be classfied as Unknown.

        Args:
            decoded_dict (dict): Dictionary containing decoded spot identities with
                keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
            unknown_index (int): The index for `Unknown` category.

        Returns:
            dict: similar to input, just replace the low probability
                ones with `Unknown`.
        """
        decoded_dict['predicted_id'][decoded_dict['probability'] < pred_prob_thresh] = unknown_index
        decoded_dict['predicted_name'][decoded_dict['probability'] < pred_prob_thresh] = 'Unknown'
        return decoded_dict

    def _rescue_errors(self,
                       decoding_dict,
                       spots_intensities_vec):
        """Rescues decoded spots assigned as `'Background'` or `'Unknown'` by if their spot
        probability values have a Hamming distance of 1 from each of the barcodes.

        Args:
            decoding_dict (dict): Dictionary containing decoded spot identities with
                keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`. This dictionary has already been processed to
                assign low probability predictions to `'Unknown'`.
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                `[num_spots, (rounds * channels)]`.
        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
        """

        ch_names = list(self.df_barcodes.columns)
        ch_names.remove('Gene')
        barcodes_array = self.df_barcodes[ch_names].values
        num_barcodes = barcodes_array.shape[0]
        barcode_len = barcodes_array.shape[1]

        predicted_ids = decoding_dict['predicted_id']
        predicted_names = decoding_dict['predicted_name']
        sources = decoding_dict['source']

        attempted = 0
        successful = 0
        for i,pred in tqdm(enumerate(predicted_names)):
            if pred in ['Background', 'Unknown']:
                attempted += 1
                dist_list = np.zeros(num_barcodes)
                for ii in range(num_barcodes):
                    dist_list[ii] = distance.hamming(np.round(spots_intensities_vec[i]),
                                                     barcodes_array[ii])
                scaled_dist_list = dist_list * barcode_len
                if 1 in scaled_dist_list:
                    successful += 1

                    new_gene = np.argwhere(scaled_dist_list == 1)[0][0]
                    # gene ids are 1-indexed
                    predicted_ids[i] = new_gene + 1
                    predicted_names[i] = self.df_barcodes['Gene'].values[new_gene]
                    sources[i] = 'error rescue'
        
        result = {
            'spot_index': decoding_dict['spot_index'],
            'predicted_id': predicted_ids,
            'predicted_name': predicted_names,
            'probability': decoding_dict['probability'],
            'source': sources
        }

        print('{} of {} rescue attempts were successful.'.format(successful, attempted))
        
        return(result)
    
    def _rescue_mixed_spots(self,
                            decoding_dict,
                            spots_intensities_vec,
                            prob_threshold=0.95):
        """Rescues decoded spots assigned as `'Background'` or `'Unknown'` by if their spot
        probability values have a Hamming distance of 1 from each of the barcodes.

        Args:
            decoding_dict (dict): Dictionary containing decoded spot identities with
                keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                `[num_spots, (rounds * channels)]`.
            df_barcodes (pd.DataFrame): Codebook, the first column is gene names (`'Gene'`),
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

        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
        """

        ch_names = list(self.df_barcodes.columns)
        ch_names.remove('Gene')
        barcodes_array = self.df_barcodes[ch_names].values
        num_barcodes = barcodes_array.shape[0]
        barcode_len = barcodes_array.shape[1]

        spot_indices = decoding_dict['spot_index']
        predicted_ids = decoding_dict['predicted_id']
        predicted_names = decoding_dict['predicted_name']
        probabilities = decoding_dict['probability']
        sources = decoding_dict['source']

        attempted = 0
        successful = 0
        for i,prob in tqdm(enumerate(probabilities)):
            if prob < prob_threshold:
                attempted += 1
                gene_id = predicted_ids[i]
                if gene_id > num_barcodes-2:
                    continue
                
                # gene ids are 1-indexed
                barcode = barcodes_array[gene_id-1]
                intensities_updated = spots_intensities_vec[i].copy()
                intensities_updated[barcode==1] = 0

                dist_list = np.zeros(num_barcodes)
                for ii in range(num_barcodes):
                    dist_list[ii] = distance.hamming(np.round(intensities_updated),
                                                     barcodes_array[ii])
                scaled_dist_list = dist_list * barcode_len
                if 1 in scaled_dist_list:
                    successful += 1
                    spot_indices = np.append(spot_indices, [spot_indices[i]])

                    new_id = np.argwhere(scaled_dist_list == 1)[0][0]
                    # gene ids are 1-indexed
                    predicted_ids = np.append(predicted_ids, [new_id + 1])

                    new_name = self.df_barcodes['Gene'].values[new_id]
                    predicted_names = np.append(predicted_names, [new_name])

                    probabilities = np.append(probabilities, [-1])

                    sources = np.append(sources, 'mixed rescue')

        result = {
            'spot_index': spot_indices,
            'predicted_id': predicted_ids,
            'predicted_name': predicted_names,
            'probability': probabilities,
            'source': sources
        }

        print('{} of {} rescue attempts were successful.'.format(successful, attempted))

        return(result)

    def _predict(self,
                 spots_intensities_vec,
                 num_iter,
                 batch_size,
                 pred_prob_thresh,
                 rescue_errors,
                 rescue_mixed):
        """Predict the gene assignment of each spot.

        Args:
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                `[num_spots, (rounds * channels)]`.
            num_iter (int): Number of iterations for training. Defaults to 500.
            batch_size (int): Size of batches for training. Defaults to 1000.
            pred_prob_thresh (float): The threshold of unknown category, within [0,1]. Defaults to 0.5.
            rescue_errors (bool): Whether to check if `'Background'`-  and `'Unknown'`-assigned
                spots have a Hamming distance of 1 to other barcodes.
            rescue_mixed (bool): Whether to check if low probability predictions are the result of
                two mixed barcodes.

        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
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
            decoding_dict, unknown_index, pred_prob_thresh=pred_prob_thresh)

        if rescue_errors:
            print('Revising errors...')
            decoding_dict_trunc = self._rescue_errors(decoding_dict_trunc,
                                                      spots_intensities_vec)
        if rescue_mixed:
            print('Correcting mixed barcodes...')
            decoding_dict_trunc = self._rescue_mixed_spots(decoding_dict_trunc,
                                                           spots_intensities_vec)
        
        return decoding_dict_trunc

    def predict(self,
                spots_intensities_vec,
                num_iter=500,
                batch_size=1000,
                pred_prob_thresh=0.95,
                rescue_errors=True,
                rescue_mixed=False):
        """Predict the gene assignment of each spot.

        Args:
            spots_intensities_vec (numpy.array): Array of spot probability values with shape
                `[num_spots, (rounds * channels)]`.
            num_iter (int): Number of iterations for training. Defaults to 500.
            batch_size (int): Size of batches for training. Defaults to 1000.
            pred_prob_thresh (float): The threshold of unknown category, within [0,1]. Defaults to 0.95.
            rescue_errors (bool): Whether to check if `'Background'`-  and `'Unknown'`-assigned
                spots have a Hamming distance of 1 to other barcodes.
            rescue_mixed (bool): Whether to check if low probability predictions are the result of
                two mixed barcodes.

        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
        """

        return self._predict(
            spots_intensities_vec=spots_intensities_vec,
            num_iter=num_iter,
            batch_size=batch_size,
            pred_prob_thresh=pred_prob_thresh,
            rescue_errors=rescue_errors,
            rescue_mixed=rescue_mixed)
