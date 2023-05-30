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

"""Tests for results utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from tensorflow.python.platform import test

from deepcell_spots.utils.results_utils import filter_results, gene_visualization


class TestResultsUtils(test.TestCase):
    
    def test_filter_results(self):
        df_spots = pd.DataFrame(
                [
                    [10, 10, 0, 0, 0.95, 0, 'A', 0, 'prediction', 0],
                    [20, 20, 1, 1, 0.95, 1, 'B', 1, 'error rescue', 1],
                    [30, 30, 2, 2, 0.95, 2, 'C', 2, 'mixed rescue', 1]
                ],
                columns=['x', 'y', 'batch_id', 'cell_id', 'probability', 'predicted_id',
                         'predicted_name', 'spot_index', 'source', 'masked']
            )
        # Test one batch id
        df_filter = filter_results(df_spots, batch_id=[0])
        self.assertEqual(len(df_filter), 1)
        self.assertEqual(len(df_spots.columns), len(df_filter.columns))

        # Test two batch ids
        df_filter = filter_results(df_spots, batch_id=[0, 1])
        self.assertEqual(len(df_filter), 2)
        self.assertEqual(len(df_spots.columns), len(df_filter.columns))

        # Test batch id ValueError
        with self.assertRaises(ValueError):
            _ = filter_results(df_spots, batch_id=0)

        # Test cell id
        df_filter = filter_results(df_spots, cell_id=[0])
        self.assertEqual(len(df_filter), 1)
        self.assertEqual(len(df_spots.columns), len(df_filter.columns))

        # Test cell id ValueError
        with self.assertRaises(ValueError):
            _ = filter_results(df_spots, cell_id=0)

        # Test gene name
        df_filter = filter_results(df_spots, gene_name=['A'])
        self.assertEqual(len(df_filter), 1)
        self.assertEqual(len(df_spots.columns), len(df_filter.columns))

        # Test gene name ValueError
        with self.assertRaises(ValueError):
            _ = filter_results(df_spots, gene_name='A')

        # Test source
        df_filter = filter_results(df_spots, source=['prediction'])
        self.assertEqual(len(df_filter), 1)
        self.assertEqual(len(df_spots.columns), len(df_filter.columns))

        # Test source ValueError
        with self.assertRaises(ValueError):
            _ = filter_results(df_spots, source='prediction')

        # Test masked
        df_filter = filter_results(df_spots, masked=True)
        self.assertEqual(len(df_filter), 1)
        self.assertEqual(len(df_spots.columns), len(df_filter.columns))


    def test_gene_visualization(self):
        df_spots = pd.DataFrame(
                [
                    [10, 10, 0, 0.95, 0, 'A', 0, 'prediction'],
                    [20, 20, 0, 0.95, 1, 'B', 1, 'error rescue'],
                    [30, 30, 0, 0.95, 2, 'C', 2, 'mixed rescue']
                ],
                columns=['x', 'y', 'batch_id', 'probability', 'predicted_id', 'predicted_name',
                         'spot_index', 'source']
            )
        gene = 'A'
        image_dim = (100, 100)

        gene_im = gene_visualization(df_spots, gene, image_dim)
        self.assertEqual(gene_im.shape, (100, 100))
        self.assertEqual(gene_im[10, 10], 1)
