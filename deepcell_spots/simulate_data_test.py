import random
import numpy as np

from tensorflow.python.platform import test

from spot_em import *

class TestSpotEM(test.TestCase):
    def test_sim_gt_clusters(self):
        num_clusters = 1000
        tp_ratio = 0.5
        gt = sim_gt_clusters(num_clusters,tp_ratio)

        self.assertEqual(len(gt),num_clusters)
        self.assertEqual(np.round(sum(gt)/len(gt),1),tp_ratio)

        # test TP ratio greater than 1
        tp_ratio = 1.1
        with self.assertRaises(AssertionError):
            gt = sim_gt_clusters(num_clusters,tp_ratio)

        # test negative TP ratio
        tp_ratio = -0.1
        with self.assertRaises(AssertionError):
            gt = sim_gt_clusters(num_clusters,tp_ratio)

    def test_sim_detections(self):
        gt_tp = np.ones(1000)
        gt_fp = np.zeros(1000)
        tpr = 0.9
        fpr = 0.1
        det_list_tp = sim_detections(gt_tp, tpr, fpr)
        det_list_fp = sim_detections(gt_fp, tpr, fpr)

        self.assertEqual(len(gt_tp),len(det_list_tp))
        self.assertEqual(len(gt_fp),len(det_list_fp))
        self.assertEqual(np.round(sum(det_list_tp)/len(det_list_tp),1),tpr)
        self.assertEqual(np.round(sum(det_list_fp)/len(det_list_fp),1),fpr)

        # test TPR greater than 1
        tpr = 1.1
        fpr = 0.1
        with self.assertRaises(AssertionError):
            det_list_tp = sim_detections(gt_tp, tpr, fpr)

        # test negative TPR
        tpr = -0.1
        fpr = 0.1
        with self.assertRaises(AssertionError):
            det_list_tp = sim_detections(gt_tp, tpr, fpr)

    def test_sim_annotators(self):
        gt_tp = np.ones(1000)
        gt_fp = np.zeros(1000)
        tpr_list = np.array([0.9,0.8,0.7]) # to show that either list or np array works
        fpr_list = [0.1,0.2,0.3]
        data_array_tp = sim_annotators(gt_tp,tpr_list,fpr_list)
        data_array_fp = sim_annotators(gt_fp,tpr_list,fpr_list)

        self.assertEqual(np.shape(data_array_tp),(len(gt_tp),len(tpr_list)))
        self.assertEqual(np.shape(data_array_fp),(len(gt_fp),len(tpr_list)))
        for i in range(len(tpr_list)):
            self.assertEqual(np.round(sum(data_array_tp[:,i])/len(data_array_tp[:,i]),1),tpr_list[i])
            self.assertEqual(np.round(sum(data_array_fp[:,i])/len(data_array_fp[:,i]),1),fpr_list[i])

        # test incorrect type for tpr and fpr
        tpr = 1.1
        fpr = 0.1
        with self.assertRaises(AssertionError):
            data_array_tp = sim_annotators(gt_tp,tpr,fpr)

        # test different lengths for tpr_list and fpr_list
        tpr_list = [0.9]*4
        fpr_list = [0.1]*3
        with self.assertRaises(AssertionError):
            data_array_tp = sim_annotators(gt_tp,tpr_list,fpr_list)

        # test TPR greater than 1
        tpr_list = [0.9,1,1.1]
        fpr_list = [0.1]*3
        with self.assertRaises(AssertionError):
            data_array_tp = sim_annotators(gt_tp,tpr_list,fpr_list)
        

test.main()