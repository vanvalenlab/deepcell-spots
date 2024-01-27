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
"""Tests for Polaris application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from tensorflow.python.platform import test

from deepcell.model_zoo import PanopticNet
from deepcell_spots.dotnet import dot_net_2D
from deepcell_spots.applications.polaris import output_to_df, Polaris


class TestPolaris(test.TestCase):
    
    def test_output_to_df(self):
        n = 100

        spots_locations = np.random.randint(0, 10, size=(1,n,2))
        spots_locations_vec = np.concatenate([np.concatenate(
            [item, [[idx_batch]] * len(item)], axis=1)
            for idx_batch, item in enumerate(spots_locations)])
        cell_id_list = np.random.randint(1, 10, size=n)
        decoding_results = {
            'probability': [1]*n,
            'predicted_ind': [0]*n,
            'predicted_name': ['code1']*n
        }
        df = output_to_df(spots_locations_vec, cell_id_list, decoding_results)
        self.assertEqual(len(df), n)
        self.assertAllEqual(
            df.columns,
            ['x', 'y', 'batch_id', 'cell_id'] + list(decoding_results.keys())
        )
        self.assertAllEqual(df['x'].values, spots_locations_vec[:,0])
        self.assertAllEqual(df['y'].values, spots_locations_vec[:,1])
        self.assertAllEqual(df['batch_id'].values, [0]*n)
        self.assertAllEqual(df['cell_id'], cell_id_list)
        self.assertAllEqual(df['probability'], decoding_results['probability'])
        self.assertAllEqual(df['predicted_ind'], decoding_results['predicted_ind'])
        self.assertAllEqual(df['predicted_name'], decoding_results['predicted_name'])


    def test_polaris_app(self):
        with self.cached_session():
            segmentation_model = PanopticNet('resnet50',
                                             input_shape=(128, 128, 1),
                                             norm_method='whole_image',
                                             num_semantic_heads=2,
                                             num_semantic_classes=[1, 1],
                                             location=True,
                                             include_top=True,
                                             lite=True,
                                             use_imagenet=False,
                                             interpolation='bilinear')
            spots_model = dot_net_2D(receptive_field=13,
                                     input_shape=(128, 128, 1),
                                     inputs=None,
                                     n_skips=3,
                                     norm_method=None,
                                     padding_mode='reflect')
            app = Polaris(segmentation_model=segmentation_model,
                          spots_model=spots_model)

            # test output shape
            shape = app.segmentation_app.model.output_shape
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 2)
            self.assertEqual(len(shape[0]), 4)
            self.assertEqual(len(shape[1]), 4)

            shape = app.spots_app.model.output_shape
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 2)
            self.assertEqual(len(shape[0]), 4)
            self.assertEqual(len(shape[1]), 4)

            # test image type error
            with self.assertRaises(ValueError):
                _ = Polaris(spots_model=spots_model, image_type='x')

            # test segmentation type error
            with self.assertRaises(ValueError):
                _ = Polaris(spots_model=spots_model, segmentation_type='x')

            # test threshold error
            app = Polaris(spots_model=spots_model)
            spots_image = np.random.rand(1, 128, 128, 1)
            with self.assertRaises(ValueError):
                _ = app.predict(spots_image=spots_image, threshold=1.1)
            with self.assertRaises(ValueError):
                _ = app.predict(spots_image=spots_image, threshold=-1.1)

            # test segmentation app error
            app = Polaris(spots_model=spots_model, segmentation_type='no segmentation')
            spots_image = np.random.rand(1, 128, 128, 1)
            segmentation_image = np.random.rand(1, 128, 128, 1)
            with self.assertRaises(ValueError):
                _ = app.predict(spots_image=spots_image,
                                segmentation_image=segmentation_image)

            # REMOVED BC OF MESMER MODEL DOWNLOAD
            # test multiplex image type
            # app = Polaris(spots_model=spots_model, image_type='multiplex',
            #               segmentation_type='mesmer')
            # self.assertIsNone(app.decoding_app)

            # df_barcodes = pd.DataFrame(
            #     [
            #         ["code1", 1, 1, 0, 0, 0, 0],
            #         ["code2", 0, 0, 1, 1, 0, 0],
            #         ["code3", 0, 0, 0, 0, 1, 1],
            #         ["code4", 1, 0, 0, 0, 1, 0],
            #         ["code5", 0, 0, 1, 0, 0, 1],
            #         ["code6", 0, 1, 0, 0, 1, 0],
            #         ["code7", 1, 0, 1, 0, 0, 0],
            #     ],
            #     columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            #     index=np.arange(7) + 1,
            # )
            # decoding_kwargs = {'df_barcodes': df_barcodes, 'rounds': 2,
            #                    'channels': 3, 'distribution': 'Relaxed Bernoulli',
            #                    'params_mode': '2*R*C'}
            # app = Polaris(spots_model=spots_model, image_type='multiplex',
            #               decoding_kwargs=decoding_kwargs)
            # self.assertIsNotNone(app.decoding_app)

            # with self.assertRaises(ValueError):
            #     spots_image = np.random.rand(1, 128, 128, 1)
            #     _ = app.predict(spots_image=spots_image)
            # with self.assertRaises(ValueError):
            #     spots_image = np.random.rand(1, 128, 128, 6)
            #     segmentation_image = np.random.rand(2, 128, 128, 1)
            #     _ = app.predict(spots_image=spots_image,
            #                     segmentation_image=segmentation_image)
            # with self.assertRaises(ValueError):
            #     spots_image = np.random.rand(1, 128, 128, 6)
            #     segmentation_image = np.random.rand(1, 128, 128, 2)
            #     _ = app.predict(spots_image=spots_image,
            #                     segmentation_image=segmentation_image)
            # with self.assertRaises(ValueError):
            #     spots_image = np.random.rand(1, 128, 128, 6)
            #     segmentation_image = np.random.rand(2, 128, 128, 1)
            #     background_image = np.random.rand(2, 128, 128, 1)
            #     _ = app.predict(spots_image=spots_image,
            #                     segmentation_image=segmentation_image)
            # with self.assertRaises(ValueError):
            #     spots_image = np.random.rand(1, 128, 128, 6)
            #     segmentation_image = np.random.rand(1, 128, 128, 2)
            #     background_image = np.random.rand(1, 128, 128, 4)
            #     _ = app.predict(spots_image=spots_image,
            #                     segmentation_image=segmentation_image)

            # # test mask rounds
            # df_barcodes = pd.DataFrame(
            #     [
            #         ["code1", 1, 1, 0, 0, 0, 0],
            #         ["code2", 0, 0, 1, 1, 0, 0],
            #         ["code3", 0, 0, 0, 0, 1, 0],
            #         ["code4", 1, 0, 0, 0, 1, 0],
            #         ["code5", 0, 0, 1, 0, 0, 0],
            #         ["code6", 0, 1, 0, 0, 1, 0],
            #         ["code7", 1, 0, 1, 0, 0, 0],
            #     ],
            #     columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            #     index=np.arange(7) + 1,
            # )
            # r = 2
            # c = 3
            # decoding_kwargs = {'df_barcodes': df_barcodes, 'rounds': r,
            #                    'channels': c, 'distribution': 'Relaxed Bernoulli',
            #                    'params_mode': '2*R*C'}
            # app = Polaris(spots_model=spots_model, image_type='multiplex',
            #               decoding_kwargs=decoding_kwargs)

            # spots_image = np.random.rand(1, 128, 128, r*c) + 1
            # segmentation_image = np.random.rand(1, 128, 128, 1)
            # background_image = np.random.rand(1, 128, 128, 1)
            # pred = app.predict(spots_image=spots_image,
            #                    segmentation_image=segmentation_image,
            #                    background_image=background_image)
            # df_results = pred[0]
            # segmentation_result = pred[1]

            # self.assertEqual(np.sum(df_results.values[:,-1]), 0)

            # # test prediction type -- singleplex
            # app = Polaris(spots_model=spots_model)
            # spots_image = np.random.rand(1, 128, 128, 1)
            # segmentation_image = np.random.rand(1, 128, 128, 1)
            # background_image = np.random.rand(1, 128, 128, 1)
            # pred = app.predict(spots_image=spots_image,
            #                    segmentation_image=segmentation_image,
            #                    background_image=background_image)
            # df_results = pred[0]
            # segmentation_result = pred[1]
            # self.assertIsInstance(df_results, pd.DataFrame)
            # self.assertIsInstance(segmentation_result, np.ndarray)
            # self.assertAllEqual(segmentation_image.shape, segmentation_result.shape)
            # self.assertAllEqual(df_results.probability, [None]*len(df_results))
            # self.assertAllEqual(df_results.predicted_id, [None]*len(df_results))
            # self.assertAllEqual(df_results.predicted_name, [None]*len(df_results))

            # # test prediction type -- multivariate Gaussian
            # df_barcodes = pd.DataFrame(
            #     [
            #         ["code1", 1, 1, 0, 0, 0, 0],
            #         ["code2", 0, 0, 1, 1, 0, 0],
            #         ["code3", 0, 0, 0, 0, 1, 1],
            #         ["code4", 1, 0, 0, 0, 1, 0],
            #         ["code5", 0, 0, 1, 0, 0, 1],
            #         ["code6", 0, 1, 0, 0, 1, 0],
            #         ["code7", 1, 0, 1, 0, 0, 0],
            #     ],
            #     columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            #     index=np.arange(7) + 1,
            # )
            # r = 2
            # c = 3
            # decoding_kwargs = {'df_barcodes': df_barcodes, 'rounds': r,
            #                    'channels': c, 'distribution': 'Gaussian'}
            # app = Polaris(spots_model=spots_model, image_type='multiplex',
            #               decoding_kwargs=decoding_kwargs)

            # spots_image = np.random.rand(1, 128, 128, r*c) + 1
            # segmentation_image = np.random.rand(1, 128, 128, 1)
            # background_image = np.random.rand(1, 128, 128, 1)
            # pred = app.predict(spots_image=spots_image,
            #                    segmentation_image=segmentation_image,
            #                    background_image=background_image)
            # df_results = pred[0]
            # segmentation_result = pred[1]
            # self.assertIsInstance(df_results, pd.DataFrame)
            # self.assertIsInstance(segmentation_result, np.ndarray)
            # self.assertAllEqual(segmentation_image.shape, segmentation_result.shape)

            # # test prediction type -- Bernoulli
            # df_barcodes = pd.DataFrame(
            #     [
            #         ["code1", 1, 1, 0, 0, 0, 0],
            #         ["code2", 0, 0, 1, 1, 0, 0],
            #         ["code3", 0, 0, 0, 0, 1, 1],
            #         ["code4", 1, 0, 0, 0, 1, 0],
            #         ["code5", 0, 0, 1, 0, 0, 1],
            #         ["code6", 0, 1, 0, 0, 1, 0],
            #         ["code7", 1, 0, 1, 0, 0, 0],
            #     ],
            #     columns=["Gene", "r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"],
            #     index=np.arange(7) + 1,
            # )
            # r = 2
            # c = 3
            # decoding_kwargs = {'df_barcodes': df_barcodes, 'rounds': r,
            #                    'channels': c, 'distribution': 'Bernoulli'}
            # app = Polaris(spots_model=spots_model, image_type='multiplex',
            #               decoding_kwargs=decoding_kwargs)

            # spots_image = np.random.rand(1, 128, 128, r*c) + 1
            # segmentation_image = np.random.rand(1, 128, 128, 1)
            # background_image = np.random.rand(1, 128, 128, 1)
            # pred = app.predict(spots_image=spots_image,
            #                    segmentation_image=segmentation_image,
            #                    background_image=background_image)
            # df_results = pred[0]
            # segmentation_result = pred[1]
            # self.assertIsInstance(df_results, pd.DataFrame)
            # self.assertIsInstance(segmentation_result, np.ndarray)
            # self.assertAllEqual(segmentation_image.shape, segmentation_result.shape)
            # self.assertAllInRange(df_results.probability, 0, 1)
