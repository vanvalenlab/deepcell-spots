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

"""Tests for spot_em"""

import numpy as np
import pandas as pd
from tensorflow.python.platform import test

from deepcell_spots.spot_em import (calc_tpr_fpr, define_edges,
                                    det_likelihood, em_spot,
                                    norm_marg_likelihood, load_coords,
                                    cluster_coords, predict_cluster_probabilities)


class TestSpotEM(test.TestCase):
    def test_calc_tpr_fpr(self):
        num_detections = 10
        gt = np.concatenate(
            (np.ones(num_detections), np.zeros(num_detections)))
        data = np.concatenate(
            (np.ones(num_detections), np.zeros(num_detections)))
        tpr, fpr = calc_tpr_fpr(gt, data)

        self.assertEqual(tpr, 1)
        self.assertEqual(fpr, 0)

        gt = np.concatenate(
            (np.ones(num_detections), np.zeros(num_detections)))
        data = np.concatenate(
            (np.zeros(num_detections), np.ones(num_detections)))
        tpr, fpr = calc_tpr_fpr(gt, data)

        self.assertEqual(tpr, 0)
        self.assertEqual(fpr, 1)

    def test_det_likelihood(self):
        cluster_data = [1, 1, 1]
        pr_list = [1, 1, 1]
        likelihood = det_likelihood(cluster_data, pr_list)

        self.assertEqual(likelihood, 1)

        pr_list = [0, 0, 0]
        likelihood = det_likelihood(cluster_data, pr_list)

        self.assertEqual(likelihood, 0)

        cluster_data = [0, 0, 0]
        pr_list = [1, 1, 1]
        likelihood = det_likelihood(cluster_data, pr_list)

        self.assertEqual(likelihood, 0)

    def test_norm_marg_likelihood(self):
        cluster_data = [1, 1, 1]
        tpr_list = [1, 1, 1]
        fpr_list = [0, 0, 0]
        prior = 1
        tp_likelihood, fp_likelihood = norm_marg_likelihood(
            cluster_data, tpr_list, fpr_list, prior)

        self.assertEqual(tp_likelihood, 1)
        self.assertEqual(fp_likelihood, 0)

    def test_em_spot(self):
        num_clusters = 10
        num_annotators = 3
        cluster_matrix = np.ones((num_clusters, num_annotators))
        tpr_list = np.ones(num_annotators)
        fpr_list = np.zeros(num_annotators)
        tp_list_new, fp_list_new, likelihood_matrix = em_spot(
            cluster_matrix, tpr_list, fpr_list)

        self.assertEqual(len(tpr_list), len(tp_list_new))
        self.assertEqual(tpr_list.all(), 1)
        self.assertEqual(len(fpr_list), len(fp_list_new))
        self.assertEqual(fpr_list.all(), 0)
        self.assertEqual(np.shape(cluster_matrix)[
                         0], np.shape(likelihood_matrix)[0])
        self.assertEqual(likelihood_matrix.all(), 1)

    def test_define_edges(self):
        # check shape of output
        num_detections = 10
        num_annotators = 3
        coords_df = pd.DataFrame(columns=['x', 'y', 'Algorithm'])
        coords_df['x'] = np.random.random_sample((num_annotators * num_detections))
        coords_df['y'] = np.random.random_sample((num_annotators * num_detections))
        alg_list = []
        for i in range(num_annotators):
            alg_list.extend([i] * num_detections)
        coords_df['Algorithm'] = alg_list

        print('Input type: {}'.format(type(coords_df)))

        threshold = 1
        A = define_edges(coords_df, threshold)

        self.assertEqual(np.shape(A)[0], np.shape(A)[
                         1], len(coords_df))

        # test two identical points
        coords_df = pd.DataFrame(columns=['x', 'y', 'Algorithm'])
        coords_df['x'] = [1, 1]
        coords_df['y'] = [1, 1]
        coords_df['Algorithm'] = [0, 1]
        threshold = 0.5
        A = define_edges(coords_df, threshold)

        self.assertEqual(np.shape(A), (len(coords_df), len(coords_df)))
        expected_output = np.zeros((2, 2))
        expected_output[0, 1] += 1
        expected_output[1, 0] += 1
        for i in range(len(coords_df)):
            for ii in range(len(coords_df)):
                self.assertEqual(A[i][ii], expected_output[i][ii])

        threshold = 0
        A = define_edges(coords_df, threshold)
        self.assertEqual(A.all(), np.zeros((2, 2)).all())

    def test_load_coords(self):
        num_detections = 10
        image_dim = 128
        coords_dict = {'A': [np.random.random_sample((num_detections, 2)) * image_dim,
                             np.random.random_sample((num_detections, 2)) * image_dim],
                       'B': [np.random.random_sample((num_detections, 2)) * image_dim,
                             np.random.random_sample((num_detections, 2)) * image_dim]}

        coords_df = load_coords(coords_dict)

        self.assertEqual(sorted(coords_df.columns),
                         sorted(['Algorithm', 'Image', 'x', 'y', 'Cluster']))
        # 10 detections * 2 images * 2 algorithms
        self.assertEqual(len(coords_df), num_detections * 4)
        self.assertEqual(sorted(list(coords_df.Algorithm.unique())), sorted(['A', 'B']))
        self.assertEqual(sorted(list(coords_df.Image.unique())), sorted([0, 1]))

    def test_cluster_coords(self):
        num_detections = 10
        image_dim = 128
        coords_dict = {'A': [np.random.random_sample((num_detections, 2)) * image_dim,
                             np.random.random_sample((num_detections, 2)) * image_dim],
                       'B': [np.random.random_sample((num_detections, 2)) * image_dim,
                             np.random.random_sample((num_detections, 2)) * image_dim]}

        coords_df = load_coords(coords_dict)
        coords_df = cluster_coords(coords_df)

        self.assertEqual(sorted(coords_df.columns),
                         sorted(['Algorithm', 'Image', 'x', 'y', 'Cluster']))
        # 10 detections * 2 images * 2 algorithms
        self.assertEqual(len(coords_df), num_detections * 4)

    def test_predict_cluster_probabilities(self):
        num_detections = 10
        image_dim = 128
        coords_dict = {'A': [np.random.random_sample((num_detections, 2)) * image_dim,
                             np.random.random_sample((num_detections, 2)) * image_dim],
                       'B': [np.random.random_sample((num_detections, 2)) * image_dim,
                             np.random.random_sample((num_detections, 2)) * image_dim]}

        coords_df = load_coords(coords_dict)
        coords_df = cluster_coords(coords_df)

        tpr_dict = {'A': 0, 'B': 0}
        fpr_dict = {'A': 0, 'B': 0}
        prob_df = predict_cluster_probabilities(coords_df, tpr_dict, fpr_dict)

        self.assertEqual(sorted(prob_df.columns),
                         sorted(['Algorithm', 'Image', 'x', 'y', 'Cluster',
                                 'Centroid_x', 'Centroid_y', 'Probability']))
        # 10 detections * 2 images * 2 algorithms
        self.assertEqual(len(coords_df), num_detections * 4)

        bad_tpr_dict = {'C': 0, 'D': 0}
        bad_fpr_dict = {'C': 0, 'D': 0}
        with self.assertRaises(NameError):
            prob_df = predict_cluster_probabilities(coords_df, bad_tpr_dict, fpr_dict)
        with self.assertRaises(NameError):
            prob_df = predict_cluster_probabilities(coords_df, tpr_dict, bad_fpr_dict)

        with self.assertRaises(ValueError):
            prob_df = predict_cluster_probabilities(coords_df, tpr_dict, fpr_dict, prior=2)


if __name__ == '__main__':
    test.main()
