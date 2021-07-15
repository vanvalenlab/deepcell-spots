import random
import numpy as np

from tensorflow.python.platform import test

from spot_em import *

class TestSpotEM(test.TestCase):
    def test_calc_tpr_fpr(self):
        num_detections = 10
        gt = np.concatenate((np.ones(num_detections),np.zeros(num_detections)))
        data = np.concatenate((np.ones(num_detections),np.zeros(num_detections)))
        tpr,fpr = calc_tpr_fpr(gt, data)

        self.assertEqual(tpr,1)
        self.assertEqual(fpr,0)

        gt = np.concatenate((np.ones(num_detections),np.zeros(num_detections)))
        data = np.concatenate((np.zeros(num_detections),np.ones(num_detections)))
        tpr,fpr = calc_tpr_fpr(gt, data)

        self.assertEqual(tpr,0)
        self.assertEqual(fpr,1)

    def test_det_likelihood(self):
        cluster_data = [1,1,1]
        pr_list = [1,1,1]
        likelihood = det_likelihood(cluster_data,pr_list)

        self.assertEqual(likelihood,1)

        pr_list = [0,0,0]
        likelihood = det_likelihood(cluster_data,pr_list)

        self.assertEqual(likelihood,0)

        cluster_data = [0,0,0]
        pr_list = [1,1,1]
        likelihood = det_likelihood(cluster_data,pr_list)

        self.assertEqual(likelihood,0)

    def test_norm_marg_likelihood(self):
        cluster_data = [1,1,1]
        tpr_list = [1,1,1]
        fpr_list = [0,0,0]
        prior = 1
        tp_likelihood,fp_likelihood = norm_marg_likelihood(cluster_data,tpr_list,fpr_list,prior)

        self.assertEqual(tp_likelihood,1)
        self.assertEqual(fp_likelihood,0)

    def test_em_spot(self):
        num_clusters = 10
        num_annotators = 3
        cluster_matrix = np.ones((num_clusters,num_annotators))
        tpr_list = np.ones(num_annotators)
        fpr_list = np.zeros(num_annotators)
        tp_list_new,fp_list_new,likelihood_matrix = em_spot(cluster_matrix,tpr_list,fpr_list)

        self.assertEqual(len(tpr_list),len(tp_list_new))
        self.assertEqual(tpr_list.all(),1)
        self.assertEqual(len(fpr_list),len(fp_list_new))
        self.assertEqual(fpr_list.all(),0) 
        self.assertEqual(np.shape(cluster_matrix)[0],np.shape(likelihood_matrix)[0])
        self.assertEqual(likelihood_matrix.all(),1)

    def test_cluster_coords(self):
        num_detections = 10
        num_annotators = 3
        num_images = 10
        image_dim = 128
        coords = np.random.random_sample((num_annotators,num_images,num_detections,2))
        image_stack = np.random.random_sample((num_images,image_dim,image_dim))
        threshold = 1
        cluster_matrix,centroid_list,coords_updated,image_stack_updated = cluster_coords(coords,image_stack,threshold)

        self.assertEqual(len(cluster_matrix),len(np.vstack(centroid_list)))
        self.assertEqual(np.shape(cluster_matrix)[1],np.shape(coords)[0])
        self.assertLessEqual(np.shape(coords_updated)[1],num_images)
        self.assertLessEqual(np.shape(image_stack_updated)[0],num_images)
        self.assertEqual(np.shape(coords_updated)[0],num_annotators)
        self.assertEqual(np.shape(coords_updated)[2],num_detections)
        self.assertEqual(np.shape(image_stack_updated)[1],image_dim)
        self.assertEqual(np.shape(image_stack_updated)[2],image_dim)

    def test_running_total_spots(self):
        num_images = 10
        num_detections = 10
        centroid_list = np.random.random_sample((num_images,num_detections,2))
        running_total = running_total_spots(centroid_list)

        self.assertEqual(len(running_total),num_images+1)

    def test_ca_to_adjacency_matrix(self):
        num_clusters = 10
        num_annotators = 3
        ca_matrix = np.ones((num_clusters,num_annotators))
        A = ca_to_adjacency_matrix(ca_matrix)

        self.assertEqual(np.shape(A)[0],np.shape(A)[1],ca_matrix[0])

    def test_define_edges(self):
        num_detections = 10
        num_annotators = 3
        coords = np.random.random_sample((num_annotators,num_detections,2))
        threshold = 1
        A = define_edges(coords,threshold)

        self.assertEqual(np.shape(A)[0],np.shape(A)[1],len(np.vstack(coords)))

        coords = np.ones((2,2))
        threshold = 0.5
        A = define_edges(coords,threshold)

        self.assertEqual(np.shape(A),(len(coords),len(coords)))
        expected_output = np.zeros((2,2))
        expected_output[0,1] += 1
        expected_output[1,0] += 1
        for i in range(len(coords)):
            for ii in range(len(coords)):
                self.assertEqual(A[i][ii],expected_output[i][ii])

        threshold = 0
        A = define_edges(coords,threshold)
        self.assertEqual(A.all(), np.zeros((2,2)).all())

    def test_consensus_coords(self):
        num_clusters = 10 #per image
        num_images = 10
        p_matrix = np.random.random_sample((num_clusters*num_images,2))
        centroid_list = np.random.random_sample((num_images,num_clusters,2))
        running_total = np.arange(0,110,10)
        y = consensus_coords(p_matrix,centroid_list,running_total)

        self.assertLessEqual(len(y),len(centroid_list))
        self.assertLessEqual(len(y),len(running_total)-1)

if __name__ == '__main__':
    test.main()