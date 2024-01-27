# Copyright 2019-2024 The Van Valen Lab at the California Institute of
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
"""Tests for SpotDecoding application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from tensorflow.python.platform import test

from deepcell_spots.applications import SpotDecoding


class TestSpotDecoding(test.TestCase):
    def test_spot_decoding_app(self):
        # test output shape
        df_barcodes1 = pd.DataFrame(
            [
                ["code1", 1, 1, 0, 0, 0, 0],
                ["code2", 0, 0, 1, 1, 0, 0],
                ["code3", 0, 0, 0, 0, 1, 1],
                ["code4", 1, 0, 0, 0, 1, 0],
                ["code5", 0, 0, 1, 0, 0, 1],
                ["code6", 0, 1, 0, 0, 1, 0],
                ["code7", 1, 0, 1, 0, 0, 0],
            ],
            columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            index=np.arange(7) + 1,
        )
        app1 = SpotDecoding(df_barcodes=df_barcodes1, rounds=2, channels=3,
                            distribution='Relaxed Bernoulli', params_mode='2*R*C')

        spots_intensities_vec1 = np.random.rand(100, 6)
        decoding_dict_trunc1 = app1.predict(
            spots_intensities_vec=spots_intensities_vec1, num_iter=20, batch_size=100
        )
        self.assertEqual(decoding_dict_trunc1["spot_index"].shape, (100,))
        self.assertEqual(decoding_dict_trunc1["probability"].shape, (100,))
        self.assertEqual(decoding_dict_trunc1["predicted_id"].shape, (100,))
        self.assertEqual(decoding_dict_trunc1["predicted_name"].shape, (100,))
        self.assertEqual(decoding_dict_trunc1["source"].shape, (100,))

        # simple functionality test
        df_barcodes2 = pd.DataFrame(
            [["code1", 0, 0, 0, 0, 0, 0], ["code2", 1, 1, 1, 1, 1, 1]],
            columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            index=np.arange(2) + 1,
        )
        app2 = SpotDecoding(df_barcodes=df_barcodes2, rounds=2, channels=3,
                            distribution='Relaxed Bernoulli', params_mode='2*R*C')

        spots_intensities_vec21 = np.ones((100, 6))
        decoding_dict_trunc21 = app2.predict(
            spots_intensities_vec=spots_intensities_vec21, num_iter=20, batch_size=100,
            pred_prob_thresh=0.5
        )
        self.assertListEqual(
            decoding_dict_trunc21["predicted_id"].tolist(), (2 * np.ones((100,))).tolist()
        )
        self.assertListEqual(decoding_dict_trunc21["predicted_name"].tolist(), ["code2"] * 100)

        spots_intensities_vec22 = np.zeros((100, 6))
        decoding_dict_trunc22 = app2.predict(
            spots_intensities_vec=spots_intensities_vec22, num_iter=20, batch_size=100,
            pred_prob_thresh=0.5
        )
        self.assertListEqual(
            decoding_dict_trunc22["predicted_id"].tolist(), np.ones((100,)).tolist()
        )
        self.assertListEqual(decoding_dict_trunc22["predicted_name"].tolist(), ["code1"] * 100)

        # test invalid codebooks
        # codebook is not a Pandas DataFrame
        df_barcodes = np.array(
            [
                ["code1", 1, 1, 0, 0, 0, 0],
                ["code2", 0, 0, 1, 1, 0, 0],
                ["code3", 0, 0, 0, 0, 1, 1],
                ["code4", 1, 0, 0, 0, 1, 0],
                ["code5", 0, 0, 1, 0, 0, 1],
                ["code6", 0, 1, 0, 0, 1, 0],
                ["code7", 1, 0, 1, 0, 0, 0],
            ])
        with self.assertRaises(TypeError):
            _ = SpotDecoding(df_barcodes=df_barcodes, rounds=2, channels=3,
                             distribution='Relaxed Bernoulli', params_mode='2*R*C')

        # incorrect column name - "Test" instead of "Gene"
        df_barcodes = pd.DataFrame(
            [
                ["code1", 1, 1, 0, 0, 0, 0],
                ["code2", 0, 0, 1, 1, 0, 0],
                ["code3", 0, 0, 0, 0, 1, 1],
                ["code4", 1, 0, 0, 0, 1, 0],
                ["code5", 0, 0, 1, 0, 0, 1],
                ["code6", 0, 1, 0, 0, 1, 0],
                ["code7", 1, 0, 1, 0, 0, 0],
            ],
            columns=["Test", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            index=np.arange(7) + 1,
        )
        with self.assertRaises(ValueError):
            _ = SpotDecoding(df_barcodes=df_barcodes, rounds=2, channels=3,
                             distribution='Relaxed Bernoulli', params_mode='2*R*C')

        # incorrect length of barcode
        with self.assertRaises(ValueError):
            _ = SpotDecoding(df_barcodes=df_barcodes, rounds=3, channels=3,
                             distribution='Relaxed Bernoulli', params_mode='2*R*C')

        # invalid barcode values
        df_barcodes = pd.DataFrame(
            [
                ["code1", 1, 2, 0, 0, 0, 0],
                ["code2", 0, 0, 1, 2, 0, 0],
                ["code3", 0, 0, 0, 0, 1, 2],
                ["code4", 1, 0, 0, 0, 2, 0],
                ["code5", 0, 0, 2, 0, 0, 1],
                ["code6", 0, 2, 0, 0, 1, 0],
                ["code7", 1, 0, 2, 0, 0, 0],
            ],
            columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            index=np.arange(7) + 1,
        )
        with self.assertRaises(ValueError):
            _ = SpotDecoding(df_barcodes=df_barcodes, rounds=2, channels=3,
                             distribution='Relaxed Bernoulli', params_mode='2*R*C')

        # invalid 'Background' codebook entry
        df_barcodes = pd.DataFrame(
            [
                ["code1", 1, 2, 0, 0, 0, 0],
                ["code2", 0, 0, 1, 2, 0, 0],
                ["code3", 0, 0, 0, 0, 1, 2],
                ["code4", 1, 0, 0, 0, 2, 0],
                ["code5", 0, 0, 2, 0, 0, 1],
                ["code6", 0, 2, 0, 0, 1, 0],
                ["Background", 0, 0, 0, 0, 0, 0],
            ],
            columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            index=np.arange(7) + 1,
        )
        with self.assertRaises(ValueError):
            _ = SpotDecoding(df_barcodes=df_barcodes, rounds=2, channels=3,
                             distribution='Relaxed Bernoulli', params_mode='2*R*C')

        # test invalid spot intensities
        # Relaxed Bernoulli
        app = SpotDecoding(df_barcodes=df_barcodes1, rounds=2, channels=3,
                           distribution='Relaxed Bernoulli', params_mode='2*R*C')

        spots_intensities_vec = np.random.rand(100, 6)
        spots_intensities_vec[0,0] = 2
        with self.assertRaises(ValueError):
            _ = app.predict(spots_intensities_vec=spots_intensities_vec, num_iter=20, batch_size=100)

        # Bernoulli
        app = SpotDecoding(df_barcodes=df_barcodes1, rounds=2, channels=3,
                           distribution='Bernoulli', params_mode='2*R*C')

        spots_intensities_vec = np.random.rand(100, 6)
        spots_intensities_vec[0,0] = 2
        with self.assertRaises(ValueError):
            _ = app.predict(spots_intensities_vec=spots_intensities_vec, num_iter=20, batch_size=100)

        spots_intensities_vec = np.random.rand(100, 6)
        spots_intensities_vec[0,0] = 0.5
        with self.assertRaises(ValueError):
            _ = app.predict(spots_intensities_vec=spots_intensities_vec, num_iter=20, batch_size=100)

        # Test mixed rescue
        app = SpotDecoding(df_barcodes=df_barcodes1, rounds=2, channels=3,
                           distribution='Relaxed Bernoulli', params_mode='2*R*C')

        spots_intensities_vec = np.random.rand(100, 6)
        decoding_dict = app.predict(spots_intensities_vec=spots_intensities_vec, num_iter=20, batch_size=100,
                                    rescue_mixed=True)
        
        self.assertGreaterEqual(decoding_dict["spot_index"].shape, (100,))
        self.assertGreaterEqual(decoding_dict["probability"].shape, (100,))
        self.assertGreaterEqual(decoding_dict["predicted_id"].shape, (100,))
        self.assertGreaterEqual(decoding_dict["predicted_name"].shape, (100,))
        self.assertGreaterEqual(decoding_dict["source"].shape, (100,))
